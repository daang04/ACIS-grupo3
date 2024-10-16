import streamlit as st
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import requests
from datetime import datetime
import io
import torch
import torch.nn.functional as F
import os
import sys

# Agregar la ruta de 'Repositorio_OG' al sys.path
sys.path.append(os.path.abspath("Repositorio_OG"))

# Ahora intenta importar UnetGenerator desde models.networks
from models.networks import UnetGenerator


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


# Clase para manejar el modelo de IA entrenado con CycleGAN
class IA_CycleGAN_Modelo:
    def __init__(self, modelo_path_G_A, modelo_path_G_B):
        # Cargar ambos generadores (A -> B y B -> A)
        try:
            self.generador_A2B = self.cargar_modelo(modelo_path_G_A)
            self.generador_B2A = self.cargar_modelo(modelo_path_G_B)
        except Execption as e:
            st.error(f"Error al cargar los modelos: {str(e)}")

    def cargar_modelo(self, modelo_path):
        try: 
            # Cargar el modelo en formato .pth
            modelo = UnetGenerator(input_nc=1, output_nc=1, num_downs=8)  # Ajusta los parámetros según lo que utilizaste
            modelo.load_state_dict(torch.load(modelo_path, map_location=torch.device('cpu')))
            modelo.eval()  # Establecer el modelo en modo evaluación
            st.success(f"Modelo {modelo_path} IA cargado correctamente.")
            return modelo
        except Exception as e:
            st.error(f"Error al cargar el modelo desde {modelo_path}: {str(e)}")
            return None

    def predecir(self, imagen, direccion):
        if imagen is not None:
            # Convertir imagen a RGB y redimensionar
            imagen = Image.fromarray(imagen).convert('RGB')
            imagen = imagen.resize((256, 256))  # Redimensionar al tamaño que espera el modelo

            # Convertir la imagen a tensor
            imagen_tensor = torch.tensor(np.array(imagen)).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalización

            # Seleccionar el generador según la dirección seleccionada
            if direccion == "A -> B":
                generador = self.generador_A2B
            else:
                generador = self.generador_B2A

            # Realizar la predicción
            with torch.no_grad():
                prediccion = generador(imagen_tensor)

            # Convertir la predicción a numpy para mostrar
            prediccion_img = prediccion.squeeze().permute(1, 2, 0).detach().numpy()  # Volver a imagen
            return prediccion_img
        else:
            st.warning("No se ha proporcionado una imagen válida para predecir.")
            return None


# Función principal que define la estructura de la aplicación con múltiples páginas
def main():
    # Cargar los modelos entrenados (generadores A->B y B->A)
    modelo_G_A2B = 'Avances/Presentacion Parcial/Modelos_entrenados/generador_A_entrenado.pth'  # Cambia esta ruta
    modelo_G_B2A = 'Avances/Presentacion Parcial/Modelos_entrenados/generador_B_entrenado.pth'  # Cambia esta ruta

    # URL en formato RAW del icono
    logo_url = "https://raw.githubusercontent.com/daang04/ACIS-grupo3/main/icon_MEDGAN%20(1).png"

    st.set_page_config(page_icon = logo_url, page_title='MEDGAN')

    st.markdown(
        """
        <style>
        .header {
            display: flex;
            align-items: center;
        }
        .header img {
            margin-right: 20px; /* Espaciado entre la imagen y el título */
        }
        </style>
        <div class="header">
            <img src="https://raw.githubusercontent.com/daang04/ACIS-grupo3/main/icon_MEDGAN%20(1).png" width="70">
            <h1 style="margin: 0;">MEDGAN</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("**Disclaimer:** Este programa ha sido diseñado con fines académicos por lo que no se recomienda el uso de las imágenes generadas como guía o resultados de algún tipo para el diagnóstico.")

    with st.sidebar:
        menu_seleccionado = option_menu("Menú", ["SCYCLE-GAN", "ACPIS", "Manual de Usuario"],
                                        icons=["cloud-upload", "bi bi-clipboard-fill","book"],
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

        st.header("Cargar Imagen (DICOM o JPG) y Predicción")

        uploaded_file = st.file_uploader("Elige un archivo DICOM o JPG", type=["dcm", "jpg", "jpeg"])

        if uploaded_file is not None:
            try:
                # Cargar los modelos de IA
                modelo_ia = IA_CycleGAN_Modelo(modelo_G_A2B, modelo_G_B2A)
            except Exception as e:
                st.error(f"Error al cargar el modelo desde {modelo_G_A2B}: {str(e)}")

            # Obtener la extensión del archivo subido
            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "dcm":
                # Procesamiento específico para archivos DICOM
                dicom_processor.cargar_archivo(uploaded_file)
                dicom_processor.mostrar_imagen()
                imagen = dicom_processor.obtener_imagen()
            else:
                # Leer el archivo como imagen
                imagen = Image.open(io.BytesIO(uploaded_file.read()))  # Lee el archivo JPG o JPEG
                flag = 1

            # Selección de la dirección de la traducción
            direccion = st.radio("Selecciona la dirección de traducción", ("A -> B", "B -> A"))

            if st.button("Realizar Predicción"):
                try:
                    prediccion = modelo_ia.predecir(imagen, direccion)
                except Exception as e:
                    prediccion = imagen
                    st.warning("No se ha podido predecir correctamente")

                if prediccion is not None:
                    st.image(prediccion, caption="Resultado de la Predicción", use_column_width=True)

                    # Guardar la imagen con el formato especificado
                    buffer = io.BytesIO()
                    prediccion_img = Image.fromarray((prediccion * 255).astype(np.uint8))
                    prediccion_img.save(buffer, format="JPEG")
                    buffer.seek(0)

                    # Definir el nombre del archivo
                    nombre_archivo = f"{nombre_paciente}_{dni_paciente}_{fecha_examen}.jpg"

                    # Botón para descargar la imagen
                    st.download_button(
                        label="Descargar Imagen",
                        data=buffer,
                        file_name=nombre_archivo,
                        mime="image/jpeg"
                    )
                else:
                    st.write('No se pudo realizar la predicción.')

    # Página 2: Personalizar (vacía)
    elif menu_seleccionado == "ACPIS":
        st.header("S-CycleGAN: Semantic Segmentation Enhanced CT-Ultrasound Image-to-Image Translation for Robotic Ultrasonography")

        github = 'https://github.com/daang04/ACIS-grupo3/blob/main/README.md'

        st.subheader('Introducción')
        st.write('El artículo aborda los desafíos en el análisis de imágenes de ultrasonido...')

    elif menu_seleccionado == "Manual de Usuario":
        st.title('Manual de Usuario')

        st.subheader('¿Para qué sirve?')
        st.write('Esta es una aplicación para transformar imágenes de CT a US y viceversa.')

        image_path = 'https://raw.githubusercontent.com/daang04/ACIS-grupo3/main/ha-removebg-preview.png'
        st.image(image_path, width=300)

        st.subheader('¿Cuáles son los parámetros empleados?')
        st.write('Los parámetros que se pueden variar son: el nombre del paciente, el DNI, la fecha, y una imagen en formato DICOM o JPEG.')


if __name__ == "__main__":
    main()
