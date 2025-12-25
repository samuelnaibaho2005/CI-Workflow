import requests
import json

# URL endpoint dari model yang sedang di-serve oleh container Docker
# Pastikan container Docker Anda berjalan dan port 5003 ter-ekspos ke host Anda.
url = "http://127.0.0.1:5004/invocations"

# Header untuk request, menandakan bahwa kita mengirim data dalam format JSON
headers = {
    "Content-Type": "application/json"
}

# Membaca data input dari file input.json
try:
    with open('input.json', 'r') as f:
        # Langsung gunakan isi dari 'dataframe_split' sebagai payload utama.
        # Ini adalah format "split" pandas, yang mungkin diharapkan oleh autolog.
        payload = json.load(f).get('dataframe_split')
        if not payload:
            raise ValueError("Format JSON tidak valid, 'dataframe_split' tidak ditemukan.")

except (FileNotFoundError, ValueError) as e:
    print(f"Error memproses 'input.json': {e}")
    print("Menggunakan data contoh sebagai gantinya.")
    # Jika file tidak ditemukan atau format salah, gunakan data contoh dalam format yang benar
    payload = {
      "columns": ["calories", "proteins", "fat", "carbohydrate"],
      "data": [
        [0.297, 0.11, 0.285, 0.0],
        [0.545, 0.285, 0.37, 0.032],
        [0.0, 0.0, 0.002, 0.0],
        [0.047, 0.013, 0.004, 0.016],
        [0.039, 0.053, 0.005, 0.005]
      ]
    }

# Mengubah payload ke format JSON string untuk dikirim dalam body request
json_data = json.dumps(payload)

# Mengirim request POST ke server
print("Mengirim data ke model di URL:", url)
try:
    response = requests.post(url, data=json_data, headers=headers)
    response.raise_for_status()  # Akan raise exception jika status code bukan 2xx

    # Mendapatkan prediksi dari response JSON
    predictions = response.json().get("predictions")

    print("\nPrediksi dari model:")
    print(predictions)

except requests.exceptions.RequestException as e:
    print(f"\nError saat melakukan request ke model: {e}")
    print("Pastikan container Docker yang menjalankan model sudah berjalan dan port 5004 dapat diakses.")

except Exception as e:
    print(f"\nTerjadi error: {e}")
