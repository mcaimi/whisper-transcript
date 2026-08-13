import sys
import os


def main():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    sys.argv = ["streamlit", "run", app_path]
    from streamlit.web.cli import main as st_main
    st_main()


if __name__ == "__main__":
    main()
