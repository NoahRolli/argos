"""ARGOS entry point. Run with:  python -m src.main"""
from .app import ArgosApp


def main() -> None:
    ArgosApp().run()


if __name__ == "__main__":
    main()