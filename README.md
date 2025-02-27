# **OcuPredict - Back-End de Diagnóstico Ocular**

Este repositorio contiene la implementación del **back-end** de **OcuPredict**, un sistema integral de diagnóstico ocular diseñado para profesionales de la salud. La aplicación se ha desarrollado con **Flask** y utiliza:

- **JWT** para la autenticación y autorización.
- **MySQL** (o el motor configurado en `config.py`) como base de datos.
- **AWS S3** para almacenar las imágenes subidas por los médicos.
- Un modelo de **PyTorch** para realizar inferencias sobre las imágenes oculares.
- **Envío de correos electrónicos** para la verificación de cuentas y recuperación de contraseñas.

El back-end expone endpoints REST para:

- **Autenticación y gestión de usuarios:** Registro, inicio de sesión, verificación por código vía correo y recuperación de contraseña.
- **Predicción:** Subida y análisis de imágenes oculares, con almacenamiento de resultados y retroalimentación.
- **Historial de solicitudes:** Consulta de análisis realizados, descarga de imágenes y envío de feedback.
- **Dashboard:** Endpoints para obtener métricas (solicitudes totales, porcentaje de recomendaciones correctas, etc.) que se consumen en el front-end.

---

## **Características del Proyecto**

- **API RESTful:** Organizada mediante Blueprints (por ejemplo, `/auth`, `/predict`, `/history`, `/home`).
- **Seguridad:** Uso de JWT para proteger rutas privadas y garantizar que sólo usuarios autenticados accedan a la información.
- **Integración con AWS S3:** Las imágenes se suben a S3 y se recuperan mediante URLs presignadas o se convierten a Base64 para su visualización.
- **Gestión del Modelo de PyTorch:** El modelo se descarga automáticamente desde S3 al iniciar la aplicación, evitando incluir el archivo .pth en el repositorio.
- **Envío de Correos:** Para verificación de cuenta y recuperación de contraseña.
- **Base de Datos:** Uso de SQLAlchemy para gestionar la información de solicitudes y usuarios.

---

## **Requisitos Previos**

- **Python 3.8+** (se recomienda usar un entorno virtual).
- **MySQL** o el motor de base de datos configurado en `config.py`.
- **Cuenta AWS:** Para la configuración de S3 y gestión del modelo.

---

## **Configuración Inicial**

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Danielrp551/sistema_ocu_predict_back.git
   cd sistema_ocu_predict_back
   ```

2. **Crear un entorno virtual e instalar dependencias:**
   ```bash
   python -m venv venv
   # En Linux/Mac:
   source venv/bin/activate
   # En Windows:
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configurar Variables de Entorno:**

   Crea un archivo `.env` (o configura las variables de entorno en tu sistema) con las siguientes variables, ajustándolas según tu entorno:
   ```env
   FLASK_APP=run.py
   FLASK_ENV=development

   # Configuración de la base de datos (ejemplo para MySQL)
   SQLALCHEMY_DATABASE_URI=mysql+pymysql://usuario:password@localhost/TA_DEEP

   # Configuración de JWT
   JWT_SECRET_KEY=tu_jwt_secret_key

   # Configuración de AWS S3 y Modelo
   AWS_ACCESS_KEY_ID=tu_access_key_id
   AWS_SECRET_ACCESS_KEY=tu_secret_access_key
   AWS_REGION=us-east-2
   AWS_S3_BUCKET=deep-ta-2025-0
   MODEL_S3_KEY=modelos/modelo_resnet18_v3_patience_15.pth
   ```
   **Importante:** Asegúrate de agregar `.env` a tu `.gitignore` para proteger tus credenciales.
   ```

---

## **Ejecución en Desarrollo**

1. **Descarga del Modelo desde S3 e Inicialización de la Aplicación:**

   El archivo `app.py` está configurado para, al arrancar, descargar el modelo desde AWS S3 si no existe localmente y luego inicializarlo. Asegúrate de que la variable `MODEL_S3_KEY` esté definida correctamente en tu archivo `.env`.

   Luego, inicia la aplicación:
   ```bash
   flask run
   ```
   La aplicación se iniciará en `http://localhost:5000` (o el puerto configurado).

2. **Debug y Logs:**

   Con `FLASK_ENV=development` obtendrás trazas detalladas en la consola en caso de error.

---

## **Estructura del Proyecto**

La estructura del repositorio es similar a la siguiente:

```
OcuPredict-backend/
├── src/
│   ├── app.py                # Archivo principal que crea la app y descarga el modelo desde S3
│   ├── config.py             # Configuración general (variables, etc.)
│   ├── controllers/          # Blueprints: auth, predict, history, home
│   │   ├── auth_controller.py
│   │   ├── predict_controller.py
│   │   ├── history_controller.py
│   │   └── home_controller.py
│   ├── models/
│   │   ├── eye_disease_model.py
│   │   └── history_model.py
│   ├── services/
│   │   ├── aws_service.py    # Funciones para interactuar con AWS S3 (upload, download, generar URL, etc.)
│   │   ├── predict_service.py
│   │   └── history_service.py
│   └── extensions.py         # Inicialización de la BD, JWT, etc.
├── requirements.txt
├── .env                    # Variables de entorno (no versionado)
├── run.py                  # Punto de entrada 
└── README.md               # Este archivo
```

---

## **Despliegue**

Este back-end se ha desplegado en **AWS EC2** (y también en Railway), lo que permite tener un entorno de producción robusto y escalable. Durante el despliegue, se configuran las variables de entorno adecuadas y se aseguran las credenciales necesarias para AWS S3 y la base de datos.

---

## **Notas Adicionales**

- **Seguridad:**  
  Las credenciales y configuraciones sensibles se manejan a través de variables de entorno, manteniéndolas fuera del repositorio público.
- **Modelo de PyTorch:**  
  El archivo del modelo se descarga automáticamente desde AWS S3 al iniciar la aplicación, evitando incluirlo en el repositorio.
- **Documentación de Endpoints:**  
  Se recomienda documentar los endpoints (por ejemplo, usando Swagger o Postman) para facilitar el consumo del API.
- **Logs y Debug:**  
  Durante el desarrollo, se activa el modo debug para obtener trazas detalladas en caso de error.

---