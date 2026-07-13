"""
Модуль логирования для СИРЕНА-КБР
=================================

Централизованная настройка логирования для всех компонентов системы.

Использование:
    from logger import get_logger
    logger = get_logger(__name__)

    logger.info("Модель обучена")
    logger.warning("Недостаточно данных")
    logger.error("Ошибка при загрузке файла")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Цветовые коды для терминала
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для терминала."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Добавляем цвет к уровню логирования
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"
        return super().format(record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Создаёт и настраивает логгер.

    Args:
        name: Имя логгера (обычно __name__)
        level: Уровень логирования (по умолчанию INFO)
        log_file: Путь к файлу логов (опционально)
        use_colors: Использовать цветной вывод в терминале

    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)

    # Не добавляем хэндлеры, если они уже есть
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Формат сообщений
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Консольный хэндлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors and sys.stdout.isatty():
        console_handler.setFormatter(ColoredFormatter(fmt, date_fmt))
    else:
        console_handler.setFormatter(logging.Formatter(fmt, date_fmt))

    logger.addHandler(console_handler)

    # Файловый хэндлер (опционально)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt, date_fmt))
        logger.addHandler(file_handler)

    return logger


def setup_root_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Настраивает корневой логгер для всего приложения.

    Args:
        level: Уровень логирования
        log_file: Путь к файлу логов
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Очищаем существующие хэндлеры
    root.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Консольный хэндлер
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter(fmt, date_fmt))
    root.addHandler(console)

    # Файловый хэндлер
    if log_file:
        file_h = logging.FileHandler(log_file, encoding='utf-8')
        file_h.setFormatter(logging.Formatter(fmt, date_fmt))
        root.addHandler(file_h)


# Пример использования
if __name__ == "__main__":
    logger = get_logger("sirena.test")

    logger.debug("Отладочное сообщение")
    logger.info("Информационное сообщение")
    logger.warning("Предупреждение")
    logger.error("Ошибка")
    logger.critical("Критическая ошибка")
