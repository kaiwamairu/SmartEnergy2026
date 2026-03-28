"""
start_ngrok.py — เปิด ngrok tunnel ไปยัง MLflow server ที่ port 5000
รัน: python scripts/start_ngrok.py
"""
import os
from pyngrok import ngrok, conf

# ใส่ authtoken จาก https://dashboard.ngrok.com/get-started/your-authtoken
# หรือ export NGROK_AUTH_TOKEN=... ใน .env ก่อนรัน
token = os.environ.get("NGROK_AUTH_TOKEN", "")
if token:
    conf.get_default().auth_token = token
else:
    print("[WARNING] NGROK_AUTH_TOKEN ไม่ได้ตั้งค่า — tunnel อาจหมดอายุเร็ว")

tunnel = ngrok.connect(5000, "http")
public_url = tunnel.public_url

print("\n" + "="*60)
print("MLflow Tracking Server พร้อมใช้งานแล้ว!")
print("="*60)
print(f"\n  Local URL  : http://localhost:5000")
print(f"  Public URL : {public_url}")
print(f"\n  → วาง URL นี้ใน Colab_Main.ipynb:")
print(f"    MLFLOW_TRACKING_URI = '{public_url}'")
print(f"\n  → Clone repo แล้วรันบน Colab:")
print(f"    !git clone https://github.com/YOUR_USERNAME/SmartEnergy2026.git")
print("\n[กด Ctrl+C เพื่อหยุด tunnel]")
print("="*60)

# รอจนกว่าจะ Ctrl+C
input("\nกด Enter เพื่อหยุด...\n")
ngrok.kill()
