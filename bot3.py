import asyncio
import csv
import io
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message,
    CallbackQuery,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    FSInputFile,
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage


# =======================
# Конфигурация
# =======================

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_IDS = {
    int(x)
    for x in os.getenv("ADMIN_IDS", "").replace(" ", "").split(",")
    if x.isdigit()
}

if not BOT_TOKEN:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в переменных окружения.")

DB_PATH = os.path.join(os.path.dirname(__file__), "attendance.db")

# Все временные значения в боте — московское время (UTC+3, без перехода на летнее время)
MSK = timezone(timedelta(hours=3), name="MSK")

# Заранее заданные здания (можно изменить координаты под свои)
BUILDINGS = {
    "A": {
        "title": "Аккуратова",
        "latitude": 60.015651,  
        "longitude": 30.303660,
        "radius_m": 80.0,
    },
    "B": {
        "title": "ИМО",
        "latitude": 60.009718,
        "longitude": 30.297507,
        "radius_m": 80.0,
    },
}


# =======================
# Работа с БД
# =======================


@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_id INTEGER UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                group_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lectures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                building_code TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                radius_m REAL NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lecture_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                accuracy REAL,
                UNIQUE (lecture_id, student_id),
                FOREIGN KEY (lecture_id) REFERENCES lectures(id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
            );
            """
        )


def get_student_by_tg_id(tg_id: int) -> Optional[Tuple]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, tg_id, full_name, group_name FROM students WHERE tg_id = ?", (tg_id,))
        return cur.fetchone()


def create_student(tg_id: int, full_name: str, group_name: str) -> None:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO students (tg_id, full_name, group_name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (tg_id, full_name.strip(), group_name.strip(), datetime.now(timezone.utc).isoformat()),
        )


def create_lecture(
    title: str,
    building_code: str,
    start_time: datetime,
    end_time: datetime,
) -> int:
    b = BUILDINGS[building_code]
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO lectures (title, building_code, start_time, end_time, latitude, longitude, radius_m)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title.strip(),
                building_code,
                start_time.isoformat(),
                end_time.isoformat(),
                b["latitude"],
                b["longitude"],
                b["radius_m"],
            ),
        )
        return cur.lastrowid


def parse_dt(value: str) -> datetime:
    """
    Парсим ISO-строку.
    - Если в БД хранится наивное время (старые записи), считаем его MSK.
    - Возвращаем timezone-aware datetime.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=MSK)
    return dt


def now_msk() -> datetime:
    return datetime.now(MSK)


def fmt_msk(dt: datetime) -> str:
    return dt.astimezone(MSK).strftime("%Y-%m-%d %H:%M")


def list_lectures(include_past: bool = True) -> List[Tuple]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, building_code, start_time, end_time FROM lectures ORDER BY start_time ASC"
        )
        rows = cur.fetchall()

    if include_past:
        return rows

    # Фильтруем в Python, т.к. строки времени могут быть с tz-offset/без него
    current = now_msk()
    filtered = []
    for r in rows:
        end_dt = parse_dt(r[4])
        if end_dt >= current:
            filtered.append(r)
    # Для студентов удобнее в хронологическом порядке
    filtered.sort(key=lambda x: parse_dt(x[3]))
    return filtered


def get_lecture(lecture_id: int) -> Optional[Tuple]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, building_code, start_time, end_time, latitude, longitude, radius_m
            FROM lectures WHERE id = ?
            """,
            (lecture_id,),
        )
        return cur.fetchone()


def add_attendance(
    lecture_id: int,
    student_id: int,
    ts: datetime,
    lat: float,
    lon: float,
    accuracy: Optional[float],
) -> bool:
    with db_conn() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO attendance (lecture_id, student_id, timestamp, latitude, longitude, accuracy)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (lecture_id, student_id, ts.isoformat(), lat, lon, accuracy),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def get_attendance_stats_csv(lecture_id: int) -> Tuple[str, bytes]:
    with db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.full_name, s.group_name, a.timestamp
            FROM attendance a
            JOIN students s ON s.id = a.student_id
            WHERE a.lecture_id = ?
            ORDER BY s.group_name, s.full_name
            """,
            (lecture_id,),
        )
        rows = cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow(["ФИО", "Группа", "Время отметки"])
    for full_name, group_name, ts in rows:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            writer.writerow([full_name, group_name, fmt_msk(dt)])
        except Exception:
            writer.writerow([full_name, group_name, ts])

    filename = f"Лекция_{lecture_id}_attendance.csv"
    return filename, output.getvalue().encode("utf-8-sig")


# =======================
# Геолокация
# =======================


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками в метрах."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def is_location_valid_for_lecture(
    lecture: Tuple,
    location,
    message_date_utc: datetime,
    message_date_msk: datetime,
) -> Tuple[bool, str]:
    """
    Минимизируем возможность обмана:
    - только в интервале лекции
    - локация достаточно точная и свежая
    - в радиусе от здания
    """
    (
        lecture_id,
        title,
        building_code,
        start_time_str,
        end_time_str,
        lat_center,
        lon_center,
        radius_m,
    ) = lecture

    start = parse_dt(start_time_str).astimezone(MSK)
    end = parse_dt(end_time_str).astimezone(MSK)

    # Разрешаем только во время лекции
    if not (start <= message_date_msk <= end):
        return False, "Отмечаться можно только во время проведения лекции."

    # Проверяем свежесть сообщения (чтобы не пересылали старую локацию)
    now_utc = datetime.now(timezone.utc)
    if abs((now_utc - message_date_utc).total_seconds()) > 120:
        return False, "Локация слишком старая. Отправьте актуальную геолокацию из Telegram."

    # Требуем live-location и проверяем, что она ещё «живая»
    live_period = getattr(location, "live_period", None)
    if live_period is None:
        return False, (
            "Нужно отправить именно live-геолокацию (живое местоположение), "
            "а не статическую точку с карты."
        )
    try:
        live_period_seconds = int(live_period)
    except (TypeError, ValueError):
        live_period_seconds = None

    if live_period_seconds is not None:
        expiry = message_date_utc + timedelta(seconds=live_period_seconds)
        # Небольшой запас в несколько секунд на задержки сети
        if now_utc > expiry + timedelta(seconds=5):
            return False, "Срок действия вашей live-геолокации истёк. Отправьте новую live-локацию."

    # Проверяем точность
    accuracy = getattr(location, "horizontal_accuracy", None)
    # Для live-локации Telegram иногда не указывает точность — в этом случае допускаем,
    # что пользователь честно транслирует своё местоположение.
    if live_period_seconds is not None and accuracy is None:
        return True, "Локация принята."

    if accuracy is None or accuracy > 100:
        return False, (
            "Точность геолокации слишком низкая. "
            "Убедитесь, что у вас включен GPS/геолокация на устройстве, и отправьте live-локацию ещё раз."
        )

    # Проверяем расстояние
    dist = haversine_distance_m(lat_center, lon_center, location.latitude, location.longitude)

    logger.info(f"Lecture {lecture_id}: "
                f"Distance={dist:.1f}m, "
                f"LivePeriod={live_period_seconds}s")
    
    if dist > radius_m:
        logger.warning(f"User пытался отметиться с расстояния {dist:.0f} м!")
        return False, f"Вы находитесь слишком далеко от здания лекции..."
    
    if dist > radius_m:
        return False, "Вы находитесь слишком далеко от здания лекции или используется поддельная локация."

    # Дополнительное требование: желательно live location
    # (у live-location в Telegram есть поле live_period на сообщении, но бот не может это жестко навязать)
    return True, "Локация принята."


# =======================
# FSM
# =======================


class RegisterStudent(StatesGroup):
    waiting_full_name = State()
    waiting_group = State()


class CreateLecture(StatesGroup):
    waiting_title = State()
    waiting_datetime = State()
    waiting_duration = State()
    waiting_building = State()


class MarkAttendance(StatesGroup):
    waiting_location = State()


# =======================
# Инициализация бота
# =======================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)


# =======================
# Утилиты
# =======================


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def main_menu_keyboard(is_student: bool, is_admin_flag: bool) -> ReplyKeyboardMarkup:
    buttons = []
    if is_student:
        buttons.append([KeyboardButton(text="📚 Предстоящие лекции")])
    if is_admin_flag:
        buttons.append([KeyboardButton(text="➕ Создать лекцию"), KeyboardButton(text="📊 Лекции")])
    return ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)


# =======================
# Обработчики команд
# =======================


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    user_id = message.from_user.id
    student = get_student_by_tg_id(user_id)
    admin_flag = is_admin(user_id)

    if not student:
        text = (
            "Привет! Это бот учета посещаемости лекций.\n\n"
            "Вы еще не зарегистрированы как студент.\n"
            "Отправьте команду /register для регистрации.\n"
        )
    else:
        text = (
            f"Здравствуйте, {student[2]}!\n"
            "Вы зарегистрированы как студент. Можете отмечаться на лекциях."
        )

    kb = main_menu_keyboard(is_student=bool(student), is_admin_flag=admin_flag)
    await message.answer(text, reply_markup=kb)
    await state.clear()


@router.message(Command("register"))
async def cmd_register(message: Message, state: FSMContext):
    student = get_student_by_tg_id(message.from_user.id)
    if student:
        await message.answer("Вы уже зарегистрированы как студент.")
        return

    await message.answer("Введите ваше ФИО полностью:")
    await state.set_state(RegisterStudent.waiting_full_name)


@router.message(RegisterStudent.waiting_full_name)
async def process_full_name(message: Message, state: FSMContext):
    full_name = message.text.strip()
    if len(full_name.split()) < 2:
        await message.answer("Пожалуйста, укажите полное ФИО (минимум фамилия и имя).")
        return

    await state.update_data(full_name=full_name)
    await message.answer("Укажите номер вашей группы (например, 101 или 606):")
    await state.set_state(RegisterStudent.waiting_group)


@router.message(RegisterStudent.waiting_group)
async def process_group(message: Message, state: FSMContext):
    group_name = message.text.strip()
    data = await state.get_data()
    full_name = data["full_name"]

    create_student(message.from_user.id, full_name, group_name)
    await state.clear()

    kb = main_menu_keyboard(is_student=True, is_admin_flag=is_admin(message.from_user.id))
    await message.answer(
        f"Регистрация завершена.\nФИО: {full_name}\nГруппа: {group_name}",
        reply_markup=kb,
    )


# =======================
# Студент: список лекций и отметка
# =======================


@router.message(F.text == "📚 Предстоящие лекции")
async def handle_upcoming_lectures(message: Message, state: FSMContext):
    student = get_student_by_tg_id(message.from_user.id)
    if not student:
        await message.answer("Сначала зарегистрируйтесь с помощью команды /register.")
        return

    lectures = list_lectures(include_past=False)
    if not lectures:
        await message.answer("Нет предстоящих лекций.")
        return

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"{row[1]} ({BUILDINGS[row[2]]['title']}, {fmt_msk(parse_dt(row[3]))})",
                    callback_data=f"lecture_select:{row[0]}",
                )
            ]
            for row in lectures
        ]
    )
    await message.answer("Выберите лекцию для отметки:", reply_markup=kb)
    await state.clear()


@router.callback_query(F.data.startswith("lecture_select:"))
async def lecture_selected(callback: CallbackQuery, state: FSMContext):
    lecture_id = int(callback.data.split(":")[1])
    lecture = get_lecture(lecture_id)
    if not lecture:
        await callback.answer("Лекция не найдена.", show_alert=True)
        return

    _, title, b_code, start_str, end_str, *_ = lecture
    b = BUILDINGS[b_code]
    text = (
        f"Лекция: {title}\n"
        f"Здание: {b['title']}\n"
        f"Время: {fmt_msk(parse_dt(start_str))} — {fmt_msk(parse_dt(end_str))}\n\n"
        "Отметиться можно только во время лекции.\n"
        "Для подтверждения присутствия:\n"
        "- включите GPS на телефоне;\n"
        "- отправьте **live location (живую геолокацию)** прямо из Telegram;\n\n"
        "Как отправить live-локацию:\n"
        "1) Нажмите на значок скрепки.\n"
        "2) Выберите «Геопозиция».\n"
        "3) Нажмите «Транслировать геопозицию» и выберите время трансляции."
    )

    location_kb = ReplyKeyboardMarkup(
        keyboard=[
            # Telegram-бот не может технически запустить live-локацию сам,
            # поэтому кнопка лишь напоминает открыть меню и отправить её вручную
            [KeyboardButton(text="📚 Предстоящие лекции")],
        ],
        
        resize_keyboard=True,
        one_time_keyboard=True,
    )

    await state.update_data(lecture_id=lecture_id)
    await state.set_state(MarkAttendance.waiting_location)
    await callback.message.answer(text, reply_markup=location_kb)
    await callback.answer()


@router.message(MarkAttendance.waiting_location, F.location)
async def handle_location(message: Message, state: FSMContext):
    data = await state.get_data()
    lecture_id = data.get("lecture_id")
    if not lecture_id:
        await message.answer("Не найдена выбранная лекция. Попробуйте снова через меню.")
        await state.clear()
        return

    lecture = get_lecture(int(lecture_id))
    if not lecture:
        await message.answer("Лекция не найдена.")
        await state.clear()
        return

    student = get_student_by_tg_id(message.from_user.id)
    if not student:
        await message.answer("Сначала зарегистрируйтесь как студент.")
        await state.clear()
        return

    ok, msg = is_location_valid_for_lecture(
        lecture=lecture,
        location=message.location,
        message_date_utc=message.date,
        message_date_msk=message.date.astimezone(MSK),
    )
    if not ok:
        await message.answer(msg)
        return

    added = add_attendance(
        lecture_id=int(lecture_id),
        student_id=student[0],
        ts=datetime.now(timezone.utc),
        lat=message.location.latitude,
        lon=message.location.longitude,
        accuracy=message.location.horizontal_accuracy,
    )
    await state.clear()

    if added:
        await message.answer("Вы успешно отметились на лекции.")
    else:
        await message.answer("Вы уже были отмечены на этой лекции.")


@router.message(MarkAttendance.waiting_location)
async def handle_no_location(message: Message):
    if message.text == "📍 Как отправить live-геолокацию":
        await message.answer(
            "Пошагово, как отправить live-геолокацию (живое местоположение):\n"
            "1) Нажмите на значок скрепки в поле ввода сообщения.\n"
            "2) Выберите пункт «Местоположение».\n"
            "3) Нажмите кнопку «Транслировать маршрут в реальном времени».\n"
            "4) Выберите время трансляции и подтвердите отправку.\n\n"
            "После этого дождитесь, пока бот подтвердит отметку."
        )
    else:
        await message.answer(
            "Пожалуйста, отправьте именно live-геолокацию (живое местоположение):\n"
            "1) Нажмите на значок скрепки.\n"
            "2) Выберите «Местоположение».\n"
            "3) Нажмите «Транслировать маршрут в реальном времени»."
        )


@router.message(F.location)
async def handle_location_without_state(message: Message, state: FSMContext):
    # Если пользователь прислал локацию вне контекста выбора лекции
    current_state = await state.get_state()
    if current_state != MarkAttendance.waiting_location:
        await message.answer(
            "Сначала выберите лекцию через кнопку «📚 Предстоящие лекции», "
            "затем следуйте инструкциям по отправке геолокации."
        )


# =======================
# Администратор: создание лекций и статистика
# =======================


@router.message(F.text == "➕ Создать лекцию")
async def admin_create_lecture(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        await message.answer("Доступ запрещен. Эта функция только для администраторов.")
        return

    await message.answer(
        "Введите название лекции (например, 'ХБП'):"
    )
    await state.set_state(CreateLecture.waiting_title)


@router.message(CreateLecture.waiting_title)
async def admin_create_lecture_title(message: Message, state: FSMContext):
    title = message.text.strip()
    await state.update_data(title=title)
    await message.answer(
        "Укажите дату и время начала лекции в формате `ГГГГ-ММ-ДД ЧЧ:MM`.\n"
        "Пример: 2026-02-15 10:30"
    )
    await state.set_state(CreateLecture.waiting_datetime)


@router.message(CreateLecture.waiting_datetime)
async def admin_create_lecture_datetime(message: Message, state: FSMContext):
    text = message.text.strip()
    try:
        start_local_naive = datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        await message.answer("Неверный формат. Используйте `ГГГГ-ММ-ДД ЧЧ:MM`, например: 2026-02-15 10:30")
        return

    start_msk = start_local_naive.replace(tzinfo=MSK)
    await state.update_data(start_time=start_msk)
    await message.answer(
        "Укажите длительность лекции в минутах (например, 90):"
    )
    await state.set_state(CreateLecture.waiting_duration)


@router.message(CreateLecture.waiting_duration)
async def admin_create_lecture_duration(message: Message, state: FSMContext):
    try:
        duration_min = int(message.text.strip())
        if duration_min <= 0 or duration_min > 300:
            raise ValueError
    except ValueError:
        await message.answer("Введите целое число минут от 1 до 300.")
        return

    data = await state.get_data()
    start_local: datetime = data["start_time"]
    end_local = start_local + timedelta(minutes=duration_min)

    # Храним как ISO со смещением, ориентируемся на MSK
    await state.update_data(start_time=start_local, end_time=end_local)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=BUILDINGS[code]["title"],
                    callback_data=f"create_lecture_building:{code}",
                )
            ]
            for code in BUILDINGS.keys()
        ]
    )
    await message.answer("Выберите здание, в котором проходит лекция:", reply_markup=kb)
    await state.set_state(CreateLecture.waiting_building)


@router.callback_query(F.data.startswith("create_lecture_building:"))
async def admin_create_lecture_building(callback: CallbackQuery, state: FSMContext):
    if not is_admin(callback.from_user.id):
        await callback.answer("Нет прав.", show_alert=True)
        return

    building_code = callback.data.split(":")[1]
    if building_code not in BUILDINGS:
        await callback.answer("Здание не найдено.", show_alert=True)
        return

    data = await state.get_data()
    title = data["title"]
    start_time: datetime = data["start_time"]
    end_time: datetime = data["end_time"]

    lecture_id = create_lecture(title, building_code, start_time, end_time)
    await state.clear()

    b = BUILDINGS[building_code]
    await callback.message.answer(
        f"Лекция создана.\n"
        f"№: {lecture_id}\n"
        f"Название: {title}\n"
        f"Здание: {b['title']}\n"
        f"Время: {fmt_msk(start_time)} — {fmt_msk(end_time)}"
    )
    await callback.answer()


@router.message(F.text == "📊 Лекции")
async def admin_lectures_list(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("Доступ запрещен.")
        return

    lectures = list_lectures(include_past=True)
    if not lectures:
        await message.answer("Лекций пока нет.")
        return

    lines = []
    for row in lectures:
        lec_id, title, b_code, start_str, end_str = row
        b = BUILDINGS.get(b_code, {"title": b_code})
        start_dt = parse_dt(start_str)
        lines.append(
            f"№ {lec_id}: {title} ({b['title']})\n{fmt_msk(start_dt)}"
        )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=f"Экспорт посещаемости для № {row[0]}",
                    callback_data=f"export_attendance:{row[0]}",
                )
            ]
            for row in lectures
        ]
    )

    await message.answer("\n\n".join(lines) + "\n\nВыберите лекцию для экспорта CSV:", reply_markup=kb)


@router.callback_query(F.data.startswith("export_attendance:"))
async def admin_export_attendance(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("Нет прав.", show_alert=True)
        return

    lecture_id = int(callback.data.split(":")[1])
    lecture = get_lecture(lecture_id)
    if not lecture:
        await callback.answer("Лекция не найдена.", show_alert=True)
        return

    filename, data = get_attendance_stats_csv(lecture_id)
    if not data or len(data) == 0:
        await callback.message.answer("По этой лекции пока нет отметившихся студентов.")
        await callback.answer()
        return

    tmp_path = os.path.join(os.path.dirname(__file__), filename)
    with open(tmp_path, "wb") as f:
        f.write(data)

    file = FSInputFile(tmp_path, filename=filename)
    _, title, *_ = lecture
    await callback.message.answer_document(
        file,
        caption=f"Статистика посещаемости лекции «{title}» (№ {lecture_id}).",
    )
    await callback.answer()


# =======================
# Точка входа
# =======================


async def main():
    init_db()
    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
