#Importar las librerias 
import time
import numpy as np
import supervision as sv
import torch
import argparse
import pandas as pd
import cv2
 
#Configuración de argumentos de línea de comandos

parser = argparse.ArgumentParser(
                    prog='yolov5',
                    description='Este programa ayudaa detectar y contar personas en regiones poligonales',
                    epilog='Texto al final de la ayuda')

parser.add_argument('-i', '--input',required=True)      # option that takes a value
parser.add_argument('-o', '--output',required=True)
parser.add_argument('-fps','--fps',type=int,default=0,help='Frames por second for output video and CSV filename')

#Guardar los fps
args = parser.parse_args()

fps_valor = args.fps

#Crear una clasepara el conteo de objetos

class CountObject():

    def __init__(self,input_video_path,output_video_path) -> None:
        # Cargar el modelo YOLOv5x6 desde el repositorio de Ultralytics
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
        
        # Obtener una paleta de colores para la representación visual
        self.colors = sv.ColorPalette.default()
        
        # Almacenar las rutas de los archivos de video de entrada y salida
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
       # Dibujar poligones de colores: verdes, amarillas, rojas y azules
        self.polygons = [
            np.array([
                [540,  985 ],
                [1620, 985 ],
                [2160, 1920],
                [1620, 2855],
                [540,  2855],
                [0,    1920]
            ], np.int32),
            np.array([
                [0,    1920],
                [540,  985 ],
                [0,    0   ]
            ], np.int32),
            np.array([
                [1620, 985 ],
                [2160, 1920],
                [2160,    0]
            ], np.int32),
            np.array([
                [540,  985 ],
                [0,    0   ],
                [2160, 0   ],
                [1620, 985 ]
            ], np.int32),
            np.array([
                [0,    1920],
                [0,    3840],
                [540,  2855]
            ], np.int32),
            np.array([
                [2160, 1920],
                [1620, 2855],
                [2160, 3840]
            ], np.int32),
            np.array([
                [1620, 2855],
                [540,  2855],
                [0,    3840],
                [2160, 3840]
            ], np.int32)
        ]
        
        
         # Obtener información sobre el video desde la ruta proporcionada
        self.video_info = sv.VideoInfo.from_video_path(input_video_path)
        
        # Crear zonas poligonales a partir de los poligones definidos y la resolución del video
        self.zones = [
            sv.PolygonZone(
                polygon=polygon, 
                frame_resolution_wh=self.video_info.resolution_wh
            )
            for polygon
            in self.polygons
        ]
        
        # Crear notadores de zona poligonal con colores y propiedades específicas

        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone, 
                color=self.colors.by_idx(index), 
                thickness=6,
                text_thickness=8,
                text_scale=4
            )
            for index, zone
            in enumerate(self.zones)
        ]


        # Crear anotadores de caja delimitadora con colores y propiedades específicas   
        self.box_annotators = [
            sv.BoxAnnotator(
                color=self.colors.by_idx(index), 
                thickness=4, 
                text_thickness=4, 
                text_scale=2
                )
            for index
            in range(len(self.polygons))
        ]
        
        self.time = 0
        self.time_records = [] 

    def process_frame(self,frame: np.ndarray, i) -> np.ndarray:
        # detect
        # Incrementar el tiempo y los registros de tiempo
        self.time +=1 
        results = self.model(frame, size=1280) # Ejecutar el modelo en el cuadro
        # Convertir las detecciones del modelo a formato Supervisely
        detections = sv.Detections.from_yolov5(results)
        # Filtrar detecciones pra clase 0 (si existe) y confianza > 0.5
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
        # Inicializar una lista para contar detecciones por poligono
        counts = [0] * len(self.polygons)
        
        # Iterar sobre las zonas, los anotadores de zona y los anotadores de caja  
        for index, (zone, zone_annotator, box_annotator) in enumerate(zip(self.zones, self.zone_annotators, self.box_annotators)):
            # Obtener una mascara que indica que detecciones cen en la zona actual
            mask = zone.trigger(detections=detections)
            # Aplicar las mascara a las detecciones para filtrarlas
            detections_filtered = detections[mask]
            # Anotar las detecciones en el cuadro usando el anotador de caja
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
            # Anotar la zona en el cuadro usando el anotador de zona
            frame = zone_annotator.annotate(scene=frame)
            # Contar las detecciones filtradas para este poligono
            counts[index] = len(detections_filtered)
        # Agregar los registros de tiempo y conteos a los registros de tiempo
        self.time_records.append([self.time] + counts)    
        # Devolver el cuadro procesado  
        return frame
    # Funció que permite modificar los frames por segundo del video
    def process_video(self):
        # Abrir el archivo de video de entrada
        cap = cv2.VideoCapture(self.input_video_path)
        
        # Tasa de cuadros deseada para el video de salida
        frame_rate = fps_valor
        
        # Obtener la tasa de cuadros original del video de entrada
        
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        # Si la tasa de cuadros no se especifica, usar la tasa original
        if frame_rate == 0:
            frame_rate = original_fps
            
        # Crear el archivo de video de salida
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),frame_rate,(frame_width, frame_height)) 
        
        while True:
            # Leer un cuadro del video de entrada
            ret, frame = cap.read()
            if not ret:
                 break
            # Procesar el cuadro usando el método 'process_frame'
            processed_frame = self.process_frame(frame,self.time)
            out.write(processed_frame)
            
            # Incrementar el tiempo automáticamente según la tasa de cuadros del video de salida
            self.time += 1
            
            if frame_rate != original_fps:
            
            	#Saltar cuadros no utilizados en el video de entrada (si es necesario)
            	cap.read()    
        
       # sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path, callback=self.process_frame)

        # Liberar los recursos de carura y escritura de video
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Crear y guardar un DataFrame con los registros de tiempo
        column_names = ['Tiempo'] + [f'Poligono {i+1}' for i in range(len(self.polygons))]
        df = pd.DataFrame(self.time_records, columns=column_names)
        df.to_csv(f'resultadosFPS{fps_valor}.csv',index=False)

if __name__ == "__main__":

    obj = CountObject(args.input,args.output)
    
    start_time = time.time() # Registrar el tiempo de inicio
    
    print("Procesando el video original...")
    obj.process_video()
    
    end_time = time.time() # Registrar el tiempo de finalización
    elapsed_time = end_time - start_time
    
    print("Procesamiento completado.")
    print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")
    
    
    
    
