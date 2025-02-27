import boto3
from botocore.config import Config
import uuid
import os
import base64


AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "my-default-bucket")

my_config = Config(
    region_name=AWS_REGION,
    signature_version='s3v4'
)

s3_client = boto3.client(
    "s3",
    config=my_config,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def download_model_from_s3(local_path, s3_key):
    """
    Descarga el archivo del modelo desde S3 a local_path si no existe.
    """

    print(f"Verificando existencia del archivo en: {local_path}")
    print(f"Resultado de os.path.exists(local_path): {os.path.exists(local_path)}")
    
    if not os.path.exists(local_path):
        print(f"Descargando modelo desde S3: {s3_key} a {local_path}")
        s3_client.download_file(AWS_S3_BUCKET, s3_key, local_path)
    else:
        print("El modelo ya existe localmente.")

def upload_file_to_s3(file_bytes, original_filename):
    """
    Sube el archivo a S3 y retorna la clave (key) generada.
    :param file_bytes: contenido binario del archivo
    :param original_filename: nombre original (p.ej. "foto.jpg")
    :return: key en S3
    """
    # Generar una clave única. Podrías usar la fecha, un UUID, etc.
    # Ejemplo simple: "imagenes/uuid4-nombre-original"
    unique_id = str(uuid.uuid4())
    key = f"imagenes/{unique_id}-{original_filename}"
    
    s3_client.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=key,
        Body=file_bytes,
        ContentType="image/jpeg",  # o el mime type correcto
    )
    
    # Retorna la clave
    return key

def get_file_as_base64(key, mime_type="image/jpeg"):
    """
    Descarga el archivo de S3 con la key especificada y lo convierte a base64,
    retornando la cadena con el formato data URI.
    """
    try:
        response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=key)
        file_bytes = response["Body"].read()
        base64_str = base64.b64encode(file_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_str}"
    except Exception as e:
        print("Error al obtener el archivo de S3 y convertir a base64:", e)
        raise e

def generate_presigned_url(key, expiration=3600):
    """
    Genera una URL presignada para leer el objeto con clave 'key' en S3,
    que expira en 'expiration' segundos (por defecto 1 hora).
    """
    try:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_S3_BUCKET, "Key": key},
            ExpiresIn=expiration,
            HttpMethod="GET",
        )
        return url
    except Exception as e:
        print("Error generating presigned URL:", e)
        raise e