# ACIS-grupo3


# S-CycleGAN: Semantic Segmentation Enhanced CT-Ultrasound Image-to-Image Translation for Robotic Ultrasonography

## Introducción![ha-removebg-preview](https://github.com/user-attachments/assets/3eb24655-aed7-4e19-a0dc-7a23ec0afa78)


El artículo aborda los desafíos en el análisis de imágenes de ultrasonido, un método no invasivo ampliamente utilizado en diagnósticos médicos. Sin embargo, la calidad de las imágenes de ultrasonido puede verse comprometida por factores como el bajo contraste y la presencia de artefactos. Para superar estas limitaciones, los autores proponen un modelo avanzado de deep learning denominado S-CycleGAN, que genera imágenes sintéticas de ultrasonido a partir de datos de tomografía computarizada (CT). Este modelo integra discriminadores semánticos dentro del marco de CycleGAN para preservar los detalles anatómicos críticos durante la transferencia de estilo de imagen.

## Desarrollo

El desarrollo del trabajo se centra en la construcción de un sistema automatizado de escaneo de ultrasonido asistido por robots (RUSS), donde el modelo S-CycleGAN se utiliza para mejorar la calidad y la precisión de las imágenes de ultrasonido generadas a partir de datos de CT. Los autores describen la arquitectura de S-CycleGAN, que incluye dos generadores y dos discriminadores, junto con redes de segmentación que actúan como discriminadores semánticos. El objetivo es transformar las imágenes de CT al estilo de ultrasonido, manteniendo la consistencia semántica y la precisión anatómica.

## Técnicas de Procesamiento de Imágenes

### CycleGAN

- **Generadores y Discriminadores**: Los generadores convierten imágenes de CT en ultrasonido y viceversa, mientras que los discriminadores tratan de diferenciar entre imágenes reales y generadas.
- **Pérdida de Consistencia Cíclica**: Asegura que la traducción de una imagen de un dominio a otro y su retorno al dominio original mantenga la imagen inicial sin cambios significativos.
  
### S-CycleGAN

- **Discriminadores Semánticos**: Incorporan redes de segmentación que analizan las imágenes generadas para garantizar que se preserven las características anatómicas esenciales.
- **Pérdida de Segmentación**: Combina la pérdida de entropía cruzada y la pérdida de Dice para mejorar la precisión en la segmentación de estructuras anatómicas.

## Conclusiones

El S-CycleGAN demuestra ser eficaz en la generación de imágenes de ultrasonido sintéticas de alta calidad que mantienen las características anatómicas críticas de las imágenes de CT. Los resultados son prometedores para su aplicación en la simulación de escaneos y el desarrollo de sistemas de ultrasonido asistidos por robots. No obstante, los autores reconocen que aún existen desafíos, como la necesidad de desarrollar métricas más adecuadas para evaluar la calidad de las imágenes generadas y mejorar el proceso de entrenamiento del modelo para maximizar el uso de los datos sintéticos.

## MEDGAN
![MEDGAN_icon](https://github.com/daang04/ACIS-grupo3/blob/main/icon_MEDGAN%20(1).png)


Por ello presentamos a [MEDGAN](https://acis-grupo3-hvbgrv8mkmp7yfhyb5ujxj.streamlit.app/) la cual es una aplicación para poder generar base de datos sinteticas por medio de las estratejias explicadas enteriormente. Permitirá al peronal del sector salud obtener imagenes de US o CT para poner en practica sus conocimientos y entranar. 
