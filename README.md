# Project_AMP_Clasification-Regresion

Introducción: Los péptidos antimicrobianos son una clase de moléculas que se 
presentan como buenos antimicrobianos y con mecanismos que evitan la 
resistencia. El diseño de péptidos que posean buena actividad puede ser complejo 
y laborioso. Por tanto, el estudio de sus relaciones cuantitativas estructura-actividad
mediante algoritmos de aprendizaje automático puede dar luces a un diseño 
racional y efectivo. 
Metodología: Se recolectó información de la actividad antimicrobiana de péptidos 
y se caracterizó su estructura mediante descriptores moleculares para diseñar 
modelos de regresión y clasificación basados en algoritmos de aprendizaje 
automático. Se evaluó la contribución de cada descriptor en los modelos generados 
mediante la determinación de su importancia relativa y, finalmente, se estimó la 
actividad antimicrobiana de nuevos péptidos. 
Resultados: Se obtuvo una base de datos estructurada de péptidos 
antimicrobianos y sus descriptores con la cual se diseñaron 56 modelos de 
aprendizaje automático. Los modelos basados en Random-Forest mostraron mejor 
desempeño y de estos, los modelos de regresión mostraron desempeños variables 
(R²=0.339-0.574), mientras que los modelos de clasificación mostraron buenos 
desempeños (MCC=0.662-0.755 y ACC=0.831-0.877). Aquellos modelos basados 
en grupos bacterianos mostraron mejor desempeño que aquellos basados en todo
el conjunto de datos. Las propiedades de los nuevos péptidos generados se 
relacionan con descriptores importantes que codifican propiedades fisicoquímicas 
como menor peso molecular, mayor carga, propensión a formar estructuras alfa￾helicoidales, menor hidrofobicidad y mayor frecuencia de aminoácidos como lisina 
y serina. 
Conclusiones: Los modelos de aprendizaje automático permitieron establecer las 
relaciones estructura-actividad de péptidos antimicrobianos. Los modelos de 
clasificación tuvieron mejor desempeño que los de regresión. Estos modelos 
permitieron hacer predicciones y se propusieron nuevos péptidos de alto potencial 
antimicrobiano. Finalmente, se obtuvo una herramienta disponible en: 
https://github.com/EliezerBonifacio/AMP_Prediction_ColabTool. 
Palabras clave: Péptidos antimicrobianos, Aprendizaje automático, Relación 
Cuantitativa Estructura-Actividad, QSAR
