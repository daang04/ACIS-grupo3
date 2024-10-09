import streamlit as st
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import requests
from datetime import datetime
import io

# Clase para cargar y procesar imágenes DICOM o JPG
class DicomProcessor:
    def __init__(self):
        self.dicom_data = None
        self.image = None

    def cargar_archivo(self, uploaded_file):
        if uploaded_file.name.endswith(".dcm"):
            self.dicom_data = pydicom.dcmread(uploaded_file)
            st.write("Información del archivo DICOM cargado:")
            st.write(self.dicom_data)
        elif uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
            self.image = Image.open(uploaded_file)
            st.write("Imagen en formato JPG cargada.")

    def obtener_imagen(self):
        if self.dicom_data:
            imagen_array = self.dicom_data.pixel_array
            return imagen_array
        elif self.image:
            return np.array(self.image)
        else:
            st.warning("No se ha cargado ningún archivo válido.")
            return None

    def mostrar_imagen(self):
        imagen = self.obtener_imagen()
        if imagen is not None:
            st.image(imagen, caption="Imagen cargada", use_column_width=True)


# Clase para manejar el modelo de IA entrenado
class IA_Modelo:
    def __init__(self, modelo_path):
        self.modelo = self.cargar_modelo(modelo_path)

    def cargar_modelo(self, modelo_path):
        with open(modelo_path, 'rb') as file:
            # Aquí deberías cargar el modelo con pickle
            # modelo = pickle.load(file)
            print('su considerable madre')
        st.success("Modelo IA cargado correctamente.")
        return modelo

    def predecir(self, imagen):
        if imagen is not None:
            # Aquí agregamos cualquier procesamiento adicional que sea necesario para el modelo.
            # prediccion = self.modelo.predict([imagen.flatten()])  # Ejemplo simple
            prediccion = 'hola'
            return prediccion
        else:
            st.warning("No se ha proporcionado una imagen válida para predecir.")
            return None


# Función principal que define la estructura de la aplicación con múltiples páginas
def main():
    # Título de la aplicación
    st.title("Aplicación de Procesamiento DICOM/JPG con IA")
    
    # Agregar el disclaimer al inicio de la aplicación
    st.info("**Disclaimer:** Este programa ha sido diseñado con fines académicos por lo que no se recomienda el uso de las imágenes generadas como guía o resultados de algún tipo para el diagnóstico.")
    
    # Crear un menú de opciones para navegar entre las páginas
    # Link de los iconos: https://icons.getbootstrap.com/icons/clipboard-fill/ 
    with st.sidebar:
        menu_seleccionado = option_menu("Menú", ["SCYCLE-GAN", "ACPIS", "Manual de Usuario","DOCKERS"],
                                        icons=["cloud-upload", "bi bi-clipboard-fill","book","book"],
                                        menu_icon="cast", default_index=0)

    # Crear instancia del procesador de imágenes
    dicom_processor = DicomProcessor()

    # Página 1: Cargar y mostrar imagen DICOM o JPG
    if menu_seleccionado == "SCYCLE-GAN":
        st.header("Datos del Paciente")
    
        # Apartado para ingresar los datos del paciente
        nombre_paciente = st.text_input("Nombre del paciente")
        dni_paciente = st.text_input("DNI del paciente")
        fecha_examen = st.date_input("Fecha del examen", value=datetime.now())
    
        st.header("Transformar a ...")
        
        # Selección de tipo de imagen
        tipo_imagen = st.radio("Selecciona el tipo de imagen", ("Imagen de Ultrasonido", "Imagen de Tomografía Computarizada"))
    
        st.header("Cargar Imagen (DICOM o JPG) y Predicción")
        
        uploaded_file = st.file_uploader("Elige un archivo DICOM o JPG", type=["dcm", "jpg", "jpeg"])
        if uploaded_file is not None:
            dicom_processor.cargar_archivo(uploaded_file)
            dicom_processor.mostrar_imagen()
    
            # Cargar el modelo y hacer predicción
            modelo_ia = IA_Modelo("modelo_entrenado.pkl")
            imagen = dicom_processor.obtener_imagen()
    
            if st.button("Realizar Predicción"):
                prediccion = modelo_ia.predecir(imagen)
                if prediccion is not None:
                    #st.write(f"Resultado de la predicción: {prediccion}")

                    st.image(prediccion, caption="Resultado de la Predicción")

                    # Guardar la imagen con el formato especificado
                    prediccion_img = Image.fromarray(prediccion)
                    buffer = io.BytesIO()
                    nombre_archivo = f"{nombre_paciente}_{dni_paciente}_{fecha_examen}.jpg"
                    prediccion_img.save(buffer, format="JPEG")
                    buffer.seek(0)
    
                    # Botón para descargar la imagen
                    st.download_button(
                        label="Descargar Imagen",
                        data=buffer,
                        file_name=nombre_archivo,
                        mime="image/jpeg"
                    )

    # Página 2: Personalizar (vacía)
    elif menu_seleccionado == "ACPIS":
        st.header("S-CycleGAN: Semantic Segmentation Enhanced CT-Ultrasound Image-to-Image Translation for Robotic Ultrasonography")

        github = 'https://github.com/daang04/ACIS-grupo3/blob/main/README.md'

        st.subheader('Introducción')

        st.write('El artículo aborda los desafíos en el análisis de imágenes de ultrasonido, un método no invasivo ampliamente \n'
                 ' utilizado en diagnósticos médicos. Sin embargo, la calidad de las imágenes de ultrasonido puede verse \n '
                 'comprometida por factores como el bajo contraste y la presencia de artefactos. Para superar estas limitaciones, \n'
                 'los autores proponen un modelo avanzado de deep learning denominado S-CycleGAN, que genera imágenes sintéticas \n'
                 'de ultrasonido a partir de datos de tomografía computarizada (CT). Este modelo integra discriminadores \n '
                 'semánticos dentro del marco de CycleGAN para preservar los detalles anatómicos críticos durante la transferencia\n'
                 ' de estilo de imagen.')

        st.subheader('Desarrollo')
        st.write('El desarrollo del trabajo se centra en la construcción de un sistema automatizado de escaneo de ultrasonido \n'
                 'asistido por robots (RUSS), donde el modelo S-CycleGAN se utiliza para mejorar la calidad y la precisión de \n'
                 'las imágenes de ultrasonido generadas a partir de datos de CT. Los autores describen la arquitectura de \n'
                 'S-CycleGAN, que incluye dos generadores y dos discriminadores, junto con redes de segmentación que actúan como\n'
                 ' discriminadores semánticos. El objetivo es transformar las imágenes de CT al estilo de ultrasonido, \n '
                 'manteniendo la consistencia semántica y la precisión anatómica.')

        st.subheader('Técnicas de Procesamiento de Imágenes')
        st.write("CycleGAN Generadores y Discriminadores: Los generadores convierten imágenes de CT en ultrasonido \n"
                 "y viceversa, mientras que los discriminadores tratan de diferenciar entre imágenes reales y generadas.\n"
                 " Pérdida de Consistencia Cíclica: Asegura que la traducción de una imagen de un dominio a otro y su \n"
                 "retorno al dominio original mantenga la imagen inicial sin cambios significativos. S-CycleGAN Discriminadores\n"
                 " Semánticos: Incorporan redes de segmentación que analizan las imágenes generadas para garantizar que se \n"
                 "preserven las características anatómicas esenciales. \n"
                 "Pérdida de Segmentación: Combina la pérdida de entropía cruzada y la pérdida de Dice para mejorar la\n"
                 " precisión en la segmentación de estructuras anatómicas.")

        st.subheader('Conclusion')
        st.write("El S-CycleGAN demuestra ser eficaz en la generación de imágenes de ultrasonido sintéticas de \n"
                 "alta calidad que mantienen las características anatómicas críticas de las imágenes de CT. \n"
                 "Los resultados son prometedores para su aplicación en la simulación de escaneos y el desarrollo\n"
                 " de sistemas de ultrasonido asistidos por robots. No obstante, los autores reconocen que aún existen\n"
                 " desafíos, como la necesidad de desarrollar métricas más adecuadas para evaluar la calidad de las\n"
                 " imágenes generadas y mejorar el proceso de entrenamiento del modelo para maximizar el uso de los\n"
                 " datos sintéticos.")
        
    elif menu_seleccionado == "Manual de Usuario":
        '''st.title('Manual de Usuario')

        # Sección: ¿Para qué sirve?
        st.subheader('¿Para qué sirve?')
        st.write('Esta es una aplicación de la cual se encarga de transformar imágenes de CT a US y viceversa')
        
        # Cargar la imagen
        image_path = 'https://raw.githubusercontent.com/daang04/ACIS-grupo3/main/ha-removebg-preview.png' # Ruta de tu imagen
        # ha-removebg-preview.png
        st.image(image_path, width=300)
        
        # Sección: ¿Cuáles son los parámetros empleados?
        st.subheader('¿Cuáles son los parámetros empleados?')
        st.write('Los parámetros que se pueden variar son los siguientes:')'''

        # Título de la presentación
        st.title("Presentación: Dockerfiles y Automatización")
        
        ## Sección 1: Introducción a Docker y la Automatización
        st.header("1. Introducción a Docker y la Automatización")
        
        st.write("""
        **Objetivo de la presentación:**
        - Entender qué es un Dockerfile.
        - Aprender cómo se crean Dockerfiles.
        - Explorar la automatización con Docker en CI/CD.
        
        **¿Qué es Docker?**
        - Plataforma de contenedorización para crear, ejecutar y gestionar aplicaciones.
        
        **¿Qué es la automatización en Docker?**
        - Integración de Docker con herramientas de CI/CD (Desarrollo Continuo e Integración Continua) para facilitar el despliegue y la actualización de aplicaciones.
        """)
        
        ## Sección 2: Conceptos Fundamentales de Docker
        st.header("2. Conceptos Fundamentales de Docker")
        
        st.write("""
        **Imágenes y Contenedores:**
        - Una *imagen* es un paquete inmutable que contiene todo el código, bibliotecas y dependencias.
        - Un *contenedor* es una instancia en ejecución de una imagen.
        
        **Registro de Imágenes (DockerHub):**
        - Almacén público o privado donde se publican y gestionan las imágenes.
        
        **Virtualización vs Contenedorización:**
        - Diferencias clave entre máquinas virtuales y contenedores.
        
        **Ventajas de Docker:**
        - Portabilidad.
        - Consistencia.
        - Escalabilidad.
        """)
        
        ## Sección 3: Definiciones Clave
        st.header("3. Definiciones Clave")
        
        st.write("""
        **Dockerfile:**
        - Un archivo de texto con instrucciones que define cómo construir una imagen de Docker.
        
        **Imagen:**
        - El resultado de construir un Dockerfile. Contiene el sistema operativo, dependencias, y código de la aplicación.
        
        **Contenedor:**
        - Una instancia en ejecución de una imagen.
        
        **CI/CD (Integración y Despliegue Continuo):**
        - Proceso de automatizar el ciclo de vida del desarrollo de software, desde la construcción hasta el despliegue en producción.
        """)
        
        ## Sección 4: Introducción al Archivo de Configuración Dockerfile
        st.header("4. Introducción al Archivo de Configuración: Dockerfile")
        
        st.write("""
        **¿Qué es un Dockerfile?**
        - Un conjunto de instrucciones que define cómo construir una imagen Docker.
        
        **Estructura básica de un Dockerfile:**
        - `FROM`: Define la imagen base.
        - `WORKDIR`: Establece el directorio de trabajo.
        - `COPY`: Copia archivos desde el host al contenedor.
        - `RUN`: Ejecuta comandos durante la construcción.
        - `EXPOSE`: Expone puertos de la aplicación.
        - `CMD`: Comando que se ejecuta cuando el contenedor inicia.
        """)
        
        st.code("""
        # Ejemplo básico de Dockerfile
        FROM node:14
        WORKDIR /app
        COPY . .
        RUN npm install
        EXPOSE 3000
        CMD ["npm", "start"]
        """, language="docker")
        
        ## Sección 5: Creación de Dockerfiles
        st.header("5. Creación de Dockerfiles")
        
        st.write("""
        **Paso a paso en la creación de un Dockerfile:**
        1. Definir la imagen base.
        2. Establecer directorio de trabajo.
        3. Copiar los archivos y dependencias.
        4. Ejecutar scripts necesarios para configurar el entorno.
        5. Exponer puertos si es necesario.
        6. Configurar comandos de inicio.
        
        **Buenas prácticas:**
        - Usar imágenes base ligeras.
        - Minimizar el número de capas en el Dockerfile.
        - Usar `.dockerignore` para evitar copiar archivos innecesarios.
        """)
        
        st.code("""
        # Dockerfile optimizado
        FROM python:3.9-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY . .
        EXPOSE 5000
        CMD ["python", "app.py"]
        """, language="docker")
        
        ## Sección 6: Variables de Entorno y Persistencia en Docker
        st.header("6. Variables de Entorno y Persistencia en Docker")
        
        st.write("""
        **Variables de Entorno:**
        - Permiten personalizar el comportamiento de las aplicaciones dentro del contenedor.
        - Se definen en el Dockerfile usando `ENV`.
        
        Ejemplo de variable de entorno:
        """)
        
        st.code("""
        ENV APP_ENV=production
        """, language="docker")
        
        st.write("""
        **Persistencia de Datos:**
        - Uso de volúmenes para persistir los datos generados en contenedores.
        
        **Tipos de volúmenes:**
        - *Volúmenes anónimos*: se crean automáticamente.
        - *Volúmenes con nombre*: especificados por el usuario.
        - *Bind mounts*: enlazan un directorio específico del host al contenedor.
        """)
        
        st.code("""
        # Ejemplo de uso de volúmenes
        docker run -v /path/del/host:/path/en/contenedor my_image
        """, language="bash")
        
        ## Sección 7: Automatización del Flujo con Docker
        st.header("7. Automatización del Flujo con Docker")
        
        st.write("""
        **Automatización en CI/CD con Docker:**
        - **Construcción automática de imágenes:** Integrar la creación de imágenes con GitHub Actions, GitLab CI o Jenkins.
        - **Despliegue automático en producción:** Usar Kubernetes o Docker Swarm para la orquestación de contenedores.
        
        **Ejemplo de CI/CD con GitHub Actions:**
        """)
        
        st.code("""
        name: Build and Push Docker Image
        on:
          push:
            branches:
              - main
        jobs:
          build:
            runs-on: ubuntu-latest
            steps:
            - name: Checkout repository
              uses: actions/checkout@v2
            - name: Build and push Docker image
              uses: docker/build-push-action@v2
              with:
                push: true
                tags: username/app:latest
        """, language="yaml")
        
        ## Sección 8: Conclusiones
        st.header("8. Conclusiones")
        
        st.write("""
        **Resumen:**
        - Los Dockerfiles son una herramienta esencial para crear imágenes de contenedores.
        - La automatización con Docker y CI/CD mejora la eficiencia en el ciclo de vida de desarrollo de software.
        - El uso adecuado de variables de entorno y volúmenes asegura flexibilidad y persistencia.
        
        **Beneficios de integrar Docker en el flujo de trabajo:**
        - Reducción de errores.
        - Mayor rapidez en la entrega de aplicaciones.
        - Entornos consistentes y reproducibles.
        """)
        
        ## Recursos adicionales
        st.header("Recursos adicionales")
        st.write("[Docker Documentation](https://docs.docker.com/)")
        st.write("Tutoriales sobre GitHub Actions y CI/CD.")
    
        


if __name__ == "__main__":
    main()
