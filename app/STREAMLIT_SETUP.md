# Streamlit — local setup

From the `shelf-intelligence-ai` project root:

1. Put your trained model in `models/best.pt`.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Start the app: `python -m streamlit run app/streamlit_app.py`
4. Open the local URL shown in the terminal (usually `http://localhost:8501`).

The app does not require the training dataset. It runs on CPU and accepts JPG, JPEG and PNG images.

For GitHub, keep `models/*.pt` ignored. If you publish the model separately, document where to download it.
