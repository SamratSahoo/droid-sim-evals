import sys; print("PYVER", sys.version.split()[0])
mods = ["av","cv2","dotenv","google.genai","huggingface_hub","scipy","tyro","websockets",
        "msgpack_numpy","openpi_client","lerobot","isaaclab_tasks"]
for m in mods:
    try: __import__(m); print("PASS", m)
    except Exception as e: print("FAIL", m, "->", type(e).__name__, str(e)[:70])
