import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app

def send_verification_email(to_email, verification_code):
    """
    Envía un correo al usuario con un código de verificación.
    """
    subject = "Código de verificación"
    sender_email = current_app.config.get("MAIL_USERNAME")
    sender_password = current_app.config.get("MAIL_PASSWORD")
    smtp_server = current_app.config.get("MAIL_SERVER")
    smtp_port = current_app.config.get("MAIL_PORT")

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = to_email

    # Cuerpo del correo (texto plano)
    text = f"Tu código de verificación es: {verification_code}"
    part1 = MIMEText(text, "plain","utf-8")
    message.attach(part1)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message.as_string())
        server.quit()
    except Exception as e:
        print("Error al enviar email:", e)
