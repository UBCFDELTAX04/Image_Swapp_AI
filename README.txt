README / Instructions
---------------------
Files generated:
- app.py : Streamlit app assembled from the notebook code cells.
- requirements.txt : Suggested Python packages to install.

How to run:
1. Create a Python virtual environment and activate it.
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows (PowerShell)
2. Install requirements:
   pip install -r requirements.txt
3. Run the Streamlit app:
   streamlit run app.py

Notes:
- The notebook's code cells were appended into the app file. The app attempts to detect common swap function names such as 'swap_faces', 'face_swap', 'run_swap', 'apply_faceswap', or 'do_swap'. If your notebook defines a function under a different name, please open 'app.py' and change the 'candidate_names' list near the bottom to include the correct function name or insert a small wrapper that calls the notebook's main routine.
- Some notebook code may rely on data files, pre-trained model files, or absolute paths. If so, copy those files into the working directory where you run the Streamlit app.
- dlib and face_recognition can be difficult to install on some platforms. If you run into errors, consider using pre-built wheels for dlib or use Mediapipe-based landmark detection instead (the notebook may already contain mediapipe code).

If you want, I can:
- Attempt to create a minimal working face-swap implementation inside app.py (this requires some choices about libraries — dlib/face_recognition vs. mediapipe).
- Modify app.py to call a specific function in your notebook if you tell me its name (I tried common names but couldn't detect one automatically).
