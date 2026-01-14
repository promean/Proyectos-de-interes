"""
medicion_pupila_exacta.py
Mide EXACTAMENTE cuánta pupila se ve: 0% (cerrado) a 100% (pupila completa visible)
"""
import cv2
import numpy as np

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("🔍 MEDICIÓN EXACTA DE PUPILA - 0% a 100%")
print("=" * 60)
print("OBJETIVO: Medir cuánta pupila se ve")
print("  - 0%: Ojos cerrados (NO se ve pupila)")
print("  - 50%: Se ve media pupila")
print("  - 100%: Se ve pupila COMPLETA")
print("=" * 60)
print("\nINSTRUCCIONES:")
print("1. Cierra COMPLETAMENTE los ojos -> debe dar 0%")
print("2. Abre los ojos NORMALMENTE -> anota el %")
print("3. Abre AL MÁXIMO los ojos -> debe dar 100%")
print("4. Presiona ESPACIO para capturar valor")
print("5. ESC para salir")
print("=" * 60)

def medir_pupila_exacta(region_ojos_gris):
    """Mide porcentaje de pupila visible (0-100%)"""
    if region_ojos_gris.size == 0:
        return 0
    
    h, w = region_ojos_gris.shape
    
    # Si la región es muy pequeña, asumir ojos cerrados
    if h < 5 or w < 10:
        return 0
    
    # ============================================
    # 1. ENCONTRAR EL PUNTO MÁS OSCURO (posible pupila)
    # ============================================
    min_val = np.min(region_ojos_gris)
    min_loc = np.where(region_ojos_gris == min_val)
    
    if len(min_loc[0]) == 0 or len(min_loc[1]) == 0:
        return 0
    
    # Tomar el primer punto más oscuro
    y_centro = min_loc[0][0]
    x_centro = min_loc[1][0]
    
    # ============================================
    # 2. ANALIZAR EN DIFERENTES DIRECCIONES
    # ============================================
    porcentajes = []
    
    # Direcciones: arriba, abajo, izquierda, derecha
    direcciones = [
        (-1, 0, "ARRIBA"),   # Arriba
        (1, 0, "ABAJO"),     # Abajo  
        (0, -1, "IZQUIERDA"), # Izquierda
        (0, 1, "DERECHA")    # Derecha
    ]
    
    for dy, dx, nombre in direcciones:
        distancia = 0
        encontro_borde = False
        
        # Empezar desde el centro y avanzar en la dirección
        y = y_centro
        x = x_centro
        
        while 0 <= y < h and 0 <= x < w:
            # Valor actual del píxel
            valor_actual = region_ojos_gris[y, x]
            
            # Si encontramos un píxel MUCHO más claro, es el borde de la pupila
            # (La pupila es oscura, el iris/parte blanca es clara)
            if valor_actual > min_val + 40:  # Umbral empírico
                encontro_borde = True
                break
            
            # Avanzar
            y += dy
            x += dx
            distancia += 1
        
        if encontro_borde:
            # Distancia máxima posible en esta dirección
            if dy != 0:  # Dirección vertical
                max_posible = h // 2
            else:  # Dirección horizontal
                max_posible = w // 2
            
            if max_posible > 0:
                porcentaje_direccion = min(100, (distancia / max_posible) * 100)
                porcentajes.append(porcentaje_direccion)
    
    # ============================================
    # 3. CALCULAR PORCENTAJE PROMEDIO
    # ============================================
    if porcentajes:
        # Tomar el promedio de las direcciones
        porcentaje_final = np.mean(porcentajes)
        
        # Si el punto más oscuro está muy cerca del borde, probablemente no es pupila
        margen_borde = 3
        if (y_centro < margen_borde or y_centro > h - margen_borde or 
            x_centro < margen_borde or x_centro > w - margen_borde):
            # Ajustar hacia abajo
            porcentaje_final *= 0.5
        
        return min(100, max(0, porcentaje_final))
    
    return 0

def dibujar_analisis_pupila(frame, x_offset, y_offset, region_ojos_gris, pupila_x, pupila_y):
    """Dibuja análisis visual de la pupila"""
    if region_ojos_gris.size == 0:
        return frame
    
    # Crear versión en color para visualización
    h, w = region_ojos_gris.shape
    if h == 0 or w == 0:
        return frame
    
    # 1. Mostrar región de ojos
    ojos_color = cv2.cvtColor(region_ojos_gris, cv2.COLOR_GRAY2BGR)
    ojos_resized = cv2.resize(ojos_color, (w*2, h*2))  # Ampliar para ver mejor
    
    # Pegar en frame
    frame[y_offset:y_offset+h*2, x_offset:x_offset+w*2] = ojos_resized
    
    # 2. Dibujar cruz en el punto más oscuro (pupila)
    if pupila_x is not None and pupila_y is not None:
        px = x_offset + pupila_x * 2
        py = y_offset + pupila_y * 2
        
        # Cruz roja en la pupila
        cv2.drawMarker(frame, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        
        # Círculo verde alrededor
        cv2.circle(frame, (px, py), 15, (0, 255, 0), 1)
    
    # 3. Etiqueta
    cv2.putText(frame, "ANALISIS PUPILA", (x_offset, y_offset-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detector de rostros
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        
        # ============================================
        # REGIÓN DE OJOS (para pupila)
        # ============================================
        # Ajustar región para centrarse mejor en los ojos
        ojos_y1, ojos_y2 = int(h * 0.30), int(h * 0.50)  # Más centrado
        ojos_x1, ojos_x2 = int(w * 0.20), int(w * 0.80)
        
        if ojos_y2 > ojos_y1 and ojos_x2 > ojos_x1:
            region_ojos = gray[y+ojos_y1:y+ojos_y2, x+ojos_x1:x+ojos_x2]
            
            # Encontrar punto más oscuro para visualización
            if region_ojos.size > 0:
                min_val = np.min(region_ojos)
                min_loc = np.where(region_ojos == min_val)
                
                if len(min_loc[0]) > 0 and len(min_loc[1]) > 0:
                    pupila_y = min_loc[0][0]
                    pupila_x = min_loc[1][0]
                else:
                    pupila_x = pupila_y = None
                
                # Medir pupila exacta
                porcentaje_pupila = medir_pupila_exacta(region_ojos)
                
                # Dibujar análisis visual
                frame = dibujar_analisis_pupila(frame, 10, 250, region_ojos, pupila_x, pupila_y)
                
                # Dibujar región ocular en el rostro
                cv2.rectangle(frame, (x+ojos_x1, y+ojos_y1), 
                            (x+ojos_x2, y+ojos_y2), (0, 255, 255), 2)
            else:
                porcentaje_pupila = 0
        else:
            porcentaje_pupila = 0
        
        # ============================================
        # REGIÓN DE BOCA (para referencia)
        # ============================================
        boca_y1, boca_y2 = int(h * 0.65), int(h * 0.90)
        boca_x1, boca_x2 = int(w * 0.25), int(w * 0.75)
        
        if boca_y2 > boca_y1 and boca_x2 > boca_x1:
            region_boca = gray[y+boca_y1:y+boca_y2, x+boca_x1:x+boca_x2]
            
            if region_boca.size > 0:
                bordes_boca = cv2.Canny(region_boca, 50, 150)
                intensidad_boca = np.mean(bordes_boca)
                
                # Dibujar región de boca
                cv2.rectangle(frame, (x+boca_x1, y+boca_y1), 
                            (x+boca_x2, y+boca_y2), (255, 0, 0), 2)
        else:
            intensidad_boca = 0
        
        # ============================================
        # MOSTRAR RESULTADOS
        # ============================================
        
        # Panel principal
        cv2.rectangle(frame, (10, 10), (400, 240), (20, 20, 40), -1)
        cv2.rectangle(frame, (10, 10), (400, 240), (100, 100, 150), 2)
        
        # Título
        cv2.putText(frame, "MEDICION EXACTA DE PUPILA", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
        
        y_pos = 65
        
        # PORCENTAJE PUPILA
        if porcentaje_pupila < 10:
            color_pupila = (0, 0, 255)  # Rojo
            estado = "CERRADO"
        elif porcentaje_pupila < 40:
            color_pupila = (0, 165, 255)  # Naranja
            estado = "SEMI-CERRADO"
        elif porcentaje_pupila < 70:
            color_pupila = (0, 255, 255)  # Amarillo
            estado = "PARCIAL"
        elif porcentaje_pupila < 90:
            color_pupila = (0, 255, 0)  # Verde
            estado = "ABIERTO"
        else:
            color_pupila = (255, 255, 0)  # Cian
            estado = "COMPLETO"
        
        cv2.putText(frame, f"PUPILA VISIBLE: {porcentaje_pupila:.0f}%", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_pupila, 2)
        y_pos += 30
        
        cv2.putText(frame, f"Estado: {estado}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_pupila, 1)
        y_pos += 30
        
        # EXPLICACIÓN
        cv2.putText(frame, "0% = Ojos cerrados", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        cv2.putText(frame, "50% = Media pupila visible", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        cv2.putText(frame, "100% = Pupila completa", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 30
        
        # BOCA (referencia)
        cv2.putText(frame, f"BOCA (ref): {intensidad_boca:.1f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        y_pos += 20
        
        # Instrucciones
        cv2.putText(frame, "ESPACIO: Capturar valor", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y_pos += 20
        
        cv2.putText(frame, "ESC: Salir", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        
        # ============================================
        # BARRA DE PROGRESO PUPILA
        # ============================================
        
        barra_x = 420
        barra_y = 60
        barra_ancho = 200
        barra_alto = 30
        
        # Fondo barra
        cv2.rectangle(frame, (barra_x, barra_y), 
                     (barra_x + barra_ancho, barra_y + barra_alto), 
                     (50, 50, 50), -1)
        
        # Progreso
        progreso = int((porcentaje_pupila / 100) * barra_ancho)
        cv2.rectangle(frame, (barra_x, barra_y), 
                     (barra_x + progreso, barra_y + barra_alto), 
                     color_pupila, -1)
        
        # Marcas
        for marca in [0, 25, 50, 75, 100]:
            x_marca = barra_x + int((marca / 100) * barra_ancho)
            cv2.line(frame, (x_marca, barra_y), 
                    (x_marca, barra_y - 5), (150, 150, 150), 1)
            cv2.putText(frame, f"{marca}%", (x_marca-10, barra_y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Borde
        cv2.rectangle(frame, (barra_x, barra_y), 
                     (barra_x + barra_ancho, barra_y + barra_alto), 
                     (150, 150, 150), 2)
        
        # Texto barra
        cv2.putText(frame, "PUPILA VISIBLE", (barra_x, barra_y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)
        
        # ============================================
        # LEYENDA VISUAL
        # ============================================
        cv2.putText(frame, "LEYENDA VISUAL:", (420, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)
        
        cv2.putText(frame, "AMARILLO: Region ojos", (420, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "AZUL: Region boca", (420, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "ROJO: Pupila detectada", (420, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "VERDE: Area analizada", (420, 205), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Mostrar frame
    cv2.imshow("MEDICION EXACTA - Pupila 0% a 100%", frame)
    
    # Controles
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == 32:  # ESPACIO
        print(f"\n📊 VALOR CAPTURADO: {porcentaje_pupila:.0f}%")
        
        if porcentaje_pupila < 10:
            print("  Estado: Ojos CERRADOS")
        elif porcentaje_pupila < 40:
            print("  Estado: Pupila PARCIAL (ojos entrecerrados)")
        elif porcentaje_pupila < 70:
            print("  Estado: Pupila VISIBLE (ojos normales)")
        elif porcentaje_pupila < 90:
            print("  Estado: Pupila BIEN VISIBLE (ojos abiertos)")
        else:
            print("  Estado: Pupila COMPLETA (ojos muy abiertos)")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("🎯 RESUMEN DE MEDICIÓN")
print("=" * 60)
print("\nVALORES QUE DEBES PROBAR:")
print("1. Ojos CERRADOS completamente: ______% (debe ser ~0%)")
print("2. Ojos entrecerrados (triste): ______%")
print("3. Ojos normales (neutral): ______%")
print("4. Ojos bien abiertos: ______%")
print("5. Ojos MUY abiertos (sorpresa): ______% (debe ser ~100%)")
print("=" * 60)
print("\n⚠️  Si no obtienes 0% o 100%, DIME:")
print("   - Qué valor obtienes con ojos CERRADOS")
print("   - Qué valor obtienes con ojos MUY ABIERTOS")
print("   - Ajustaremos la fórmula para que funcione")