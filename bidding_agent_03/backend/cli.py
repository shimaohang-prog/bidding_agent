"""管理命令：python -m backend.cli create-user USERNAME。"""

import argparse
import asyncio
import getpass

from backend.core.config import get_settings
from backend.core.security import hash_password
from backend.db.session import create_engine_and_sessionmaker
from backend.repositories.users import UserRepository


async def create_user(username: str) -> None:
    password = getpass.getpass("密码（至少 8 位）：")
    if len(password) < 8:
        raise ValueError("密码至少 8 位")
    settings = get_settings()
    engine, factory = create_engine_and_sessionmaker(settings.database_url)
    try:
        async with factory() as session:
            repo = UserRepository(session)
            if await repo.by_username(username):
                raise ValueError("用户已存在")
            await repo.create(username, hash_password(password))
            await session.commit()
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-user")
    create.add_argument("username")
    args = parser.parse_args()
    if args.command == "create-user":
        asyncio.run(create_user(args.username.strip().lower()))


if __name__ == "__main__":
    main()
