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
    # URL en formato RAW del icono
    logo_url = "https://raw.githubusercontent.com/daang04/ACIS-grupo3/main/icon_MEDGAN.png"
    # Descargar la imagen
    response = requests.get(logo_url)
    img = Image.open(BytesIO(response.content))
    
    # Redimensionar a 16x16
    img_resized = img.resize((16, 16))
    
    # Guardar la imagen redimensionada si deseas
    img_resized.save("favicon_16x16.png")
    
    st.set_page_config(page_icon = img_resized)
    # favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) 
    #or a string (url or local file path)
    
    # Título de la aplicación
    st.title("MEDGAN")
    
    # Agregar el disclaimer al inicio de la aplicación
    st.info("**Disclaimer:** Este programa ha sido diseñado con fines académicos por lo que no se recomienda el uso de las imágenes generadas como guía o resultados de algún tipo para el diagnóstico.")
    
    # Crear un menú de opciones para navegar entre las páginas
    # Link de los iconos: https://icons.getbootstrap.com/icons/clipboard-fill/ 
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
    
        #st.header("Transformar a ...")
        
        # Selección de tipo de imagen
        #tipo_imagen = st.radio("Selecciona el tipo de imagen", ("Imagen de Ultrasonido", "Imagen de Tomografía Computarizada"))
        
    
        st.header("Cargar Imagen (DICOM o JPG) y Predicción")
        
        uploaded_file = st.file_uploader("Elige un archivo DICOM o JPG", type=["dcm", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Obtener la extensión del archivo subido
            file_extension = uploaded_file.name.split(".")[-1].lower()
        
            if file_extension == "dcm":
                # Procesamiento específico para archivos DICOM
                dicom_processor.cargar_archivo(uploaded_file)
                dicom_processor.mostrar_imagen()
                
                # Cargar el modelo y hacer predicción
                modelo_ia = IA_Modelo("modelo_entrenado.pkl")
                imagen = dicom_processor.obtener_imagen()
            else:
                imagen = uploaded_file
        
            if st.button("Realizar Predicción"):
                prediccion = modelo_ia.predecir(imagen)
                if prediccion is not None:
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


        # url = st.text_input('Link de la carpeta drive donde desee descargar') # permitir drive abrir
          

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
        
        st.subheader('Referencias')
        st.write('Paper base: https://arxiv.org/html/2406.01191v2')
        st.write('Repositorio referencial: https://github.com/yhsong98/ct-us-i2i-translation')
        st.write('Dataset sugerido: https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm/data')
        
        
    elif menu_seleccionado == "Manual de Usuario":
        st.title('Manual de Usuario')

        # Sección: ¿Para qué sirve?
        st.subheader('¿Para qué sirve?')
        st.write('Esta es una aplicación de la cual se encarga de transformar imágenes de CT a US y viceversa')
        
        # Cargar la imagen
        image_path = 'https://raw.githubusercontent.com/daang04/ACIS-grupo3/main/ha-removebg-preview.png' # Ruta de tu imagen
        # ha-removebg-preview.png
        st.image(image_path, width=300)
        
        # Sección: ¿Cuáles son los parámetros empleados?
        st.subheader('¿Cuáles son los parámetros empleados?')
        st.write('Los parámetros que se pueden variar son los siguientes:')

if __name__ == "__main__":
    main()
