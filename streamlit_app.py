"""
Root entrypoint for Streamlit Cloud.

This imports and runs the main() function from app/app.py so the app can be
launched using `streamlit run streamlit_app.py` or by setting this file as the
Main file on Streamlit Community Cloud.
"""

from app.app import main


if __name__ == "__main__":
    main()
