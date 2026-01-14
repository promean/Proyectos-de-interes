"""
DETECTOR DE EXPRESIONES FACIALES - VERSIÓN FINAL FUNCIONAL
Sin errores de orden
"""
import cv2
import pygame
import numpy as np
import os
import sys
from pathlib import Path
import time

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
class DetectorExpresiones:
    def __init__(self):
        print("🔧 Inicializando detector...")
        
        # PRIMERO definir colores (esto es lo que faltaba)
        self.colores = {
            'feliz': (255, 255, 0),      # Amarillo
            'triste': (0, 150, 255),     # Azul claro
            'sorpresa': (255, 0, 255),   # Magenta
            'neutral': (200, 200, 200),  # Gris
            'enojado': (255, 50, 50)     # Rojo
        }
        
        # Obtener ruta actual
        self.ruta_base = Path(__file__).parent
        print(f"📂 Carpeta del proyecto: {self.ruta_base}")
        
        # Configurar PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((1100, 700))
        pygame.display.set_caption("🎭 Detector de Expresiones - VS Code")
        
        # Fuentes
        self.font_grande = pygame.font.SysFont('Arial', 40, bold=True)
        self.font_mediana = pygame.font.SysFont('Arial', 28)
        self.font_chica = pygame.font.SysFont('Arial', 22)
        
        # Cargar imágenes
        self.imagenes_expresiones = self.cargar_imagenes_seguro()
        
        # Iniciar cámara web
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ ERROR: No se encontró cámara web")
            # Intentar con índice diferente
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("❌ No hay cámaras disponibles")
                sys.exit()
        
        # Configurar cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Cargar detector de rostros
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Variables de estado
        self.expresion_actual = "neutral"
        self.ejecutando = True
        self.mostrar_debug = False
        self.ultimo_cambio = time.time()
        self.confianza = 0.0
        self.rostro_detectado = False
        
        print("✅ ¡Sistema listo!")
    
    def crear_imagen_alternativa(self, expresion, tamano=(280, 280)):
        """Crea una imagen alternativa si no se puede cargar la original"""
        img = pygame.Surface(tamano)
        color = self.colores.get(expresion, (100, 100, 100))
        img.fill(color)
        
        # Añadir texto y emoji
        font = pygame.font.SysFont('Arial', 72)
        emojis = {
            'feliz': '😊',
            'triste': '😢',
            'sorpresa': '😲',
            'neutral': '😐',
            'enojado': '😠'
        }
        emoji = emojis.get(expresion, '😀')
        texto_emoji = font.render(emoji, True, (255, 255, 255))
        texto_rect = texto_emoji.get_rect(center=(tamano[0]//2, tamano[1]//2 - 30))
        img.blit(texto_emoji, texto_rect)
        
        font_nombre = pygame.font.SysFont('Arial', 36, bold=True)
        nombre = font_nombre.render(expresion.upper(), True, (255, 255, 255))
        nombre_rect = nombre.get_rect(center=(tamano[0]//2, tamano[1]//2 + 50))
        img.blit(nombre, nombre_rect)
        
        return img
    
    def cargar_imagenes_seguro(self):
        """Carga imágenes con múltiples intentos y formatos"""
        imagenes = {}
        expresiones = ['feliz', 'triste', 'sorpresa', 'neutral', 'enojado']
        
        for expresion in expresiones:
            cargada = False
            
            # Intentar diferentes formatos
            formatos = ['.jpg', '.jpeg', '.png', '.bmp']
            for fmt in formatos:
                ruta = self.ruta_base / "imagenes" / f"{expresion}{fmt}"
                if ruta.exists():
                    try:
                        print(f"🔄 Intentando cargar: {ruta.name}")
                        img = pygame.image.load(str(ruta))
                        img = pygame.transform.scale(img, (280, 280))
                        imagenes[expresion] = img
                        print(f"✅ Cargada: {ruta.name}")
                        cargada = True
                        break
                    except pygame.error as e:
                        print(f"⚠️  Error con {ruta.name}: {e}")
                        continue
            
            if not cargada:
                print(f"📝 Creando imagen alternativa para: {expresion}")
                imagenes[expresion] = self.crear_imagen_alternativa(expresion)
        
        return imagenes
    
    def analizar_expresion(self, rostro_gris):
        """Analiza expresión facial de manera simple pero efectiva"""
        h, w = rostro_gris.shape
        
        if h < 50 or w < 50:
            return "neutral", 0.5
        
        try:
            # 1. Analizar región de la boca
            boca_y1, boca_y2 = int(h * 0.65), int(h * 0.9)
            boca_x1, boca_x2 = int(w * 0.25), int(w * 0.75)
            
            if boca_y2 > boca_y1 and boca_x2 > boca_x1:
                region_boca = rostro_gris[boca_y1:boca_y2, boca_x1:boca_x2]
                
                # Calcular brillo promedio de la boca
                brillo_boca = np.mean(region_boca)
                
                # Detectar bordes en la boca
                bordes_boca = cv2.Canny(region_boca, 50, 150)
                intensidad_boca = np.mean(bordes_boca)
                
                # 2. Analizar región de ojos
                ojos_y1, ojos_y2 = int(h * 0.2), int(h * 0.5)
                ojos_x1, ojos_x2 = int(w * 0.15), int(w * 0.85)
                
                region_ojos = rostro_gris[ojos_y1:ojos_y2, ojos_x1:ojos_x2]
                brillo_ojos = np.mean(region_ojos)
                
                # 3. Lógica mejorada de detección
                # SORPRESA: Boca muy activa (muchos bordes)
                if intensidad_boca > 40:
                    return "sorpresa", min(0.9, intensidad_boca / 100)
                
                # FELIZ: Boca moderadamente activa, ojos normales
                elif intensidad_boca > 20 and brillo_ojos > 80:
                    return "feliz", min(0.8, intensidad_boca / 80)
                
                # ENOJADO: Ojos oscuros (entrecerrados), boca inactiva
                elif brillo_ojos < 70 and intensidad_boca < 15:
                    return "enojado", 0.7
                
                # TRISTE: Todo muy oscuro/inactivo
                elif brillo_boca < 80 and brillo_ojos < 80:
                    return "triste", 0.6
                
                # NEUTRAL: Por defecto
                else:
                    return "neutral", 0.5
                    
            else:
                return "neutral", 0.3
                
        except Exception as e:
            if self.mostrar_debug:
                print(f"Error en análisis: {e}")
            return "neutral", 0.1
    
    def dibujar_interfaz(self, frame_pygame):
        """Dibuja la interfaz gráfica"""
        # Fondo
        self.screen.fill((25, 30, 40))
        
        # ===== TÍTULO =====
        titulo = self.font_grande.render("😊 DETECTOR DE EXPRESIONES", True, (100, 220, 255))
        self.screen.blit(titulo, (50, 20))
        
        # ===== VIDEO EN VIVO =====
        # Marco del video
        pygame.draw.rect(self.screen, (40, 45, 60), (40, 80, 660, 500), 0, 10)
        pygame.draw.rect(self.screen, (80, 180, 255), (40, 80, 660, 500), 3, 10)
        
        # Video de la cámara
        self.screen.blit(frame_pygame, (50, 90))
        
        # Indicador de estado
        estado_color = (0, 255, 0) if self.rostro_detectado else (255, 100, 100)
        estado_texto = "✓ ROSTRO DETECTADO" if self.rostro_detectado else "BUSCANDO ROSTRO..."
        estado = self.font_mediana.render(estado_texto, True, estado_color)
        self.screen.blit(estado, (60, 600))
        
        # ===== EXPRESIÓN DETECTADA =====
        # Marco de expresión
        pygame.draw.rect(self.screen, (35, 40, 55), (720, 80, 350, 350), 0, 15)
        color_borde = self.colores.get(self.expresion_actual, (200, 200, 200))
        pygame.draw.rect(self.screen, color_borde, (720, 80, 350, 350), 4, 15)
        
        # Imagen de la expresión
        if self.expresion_actual in self.imagenes_expresiones:
            img_exp = self.imagenes_expresiones[self.expresion_actual]
            self.screen.blit(img_exp, (740, 100))
        
        # Nombre de la expresión
        nombre = self.font_grande.render(self.expresion_actual.upper(), True, (255, 255, 200))
        nombre_rect = nombre.get_rect(center=(895, 390))
        self.screen.blit(nombre, nombre_rect)
        
        # Confianza
        conf_texto = self.font_chica.render(f"Confianza: {self.confianza*100:.0f}%", True, (200, 200, 200))
        self.screen.blit(conf_texto, (780, 420))
        
        # Barra de confianza
        barra_ancho = 280
        barra_x = 760
        barra_y = 450
        
        # Fondo de la barra
        pygame.draw.rect(self.screen, (50, 50, 70), (barra_x, barra_y, barra_ancho, 20), 0, 10)
        
        # Barra de progreso
        progreso = int(barra_ancho * self.confianza)
        color_barra = (
            int(255 * (1 - self.confianza)),
            int(255 * self.confianza),
            100
        )
        pygame.draw.rect(self.screen, color_barra, (barra_x, barra_y, progreso, 20), 0, 10)
        
        # ===== INSTRUCCIONES =====
        instrucciones = [
            "🎮 CONTROLES:",
            "• ESPACIO: Pausar/Reanudar",
            "• ESC: Salir del programa",
            "",
            "💡 PARA MEJOR DETECCIÓN:",
            "1. Buena iluminación frontal",
            "2. Haz expresiones exageradas",
            "3. Mantén el rostro centrado",
            "4. Sonríe ampliamente para FELIZ",
            "5. Abre la boca para SORPRESA"
        ]
        
        y_pos = 500
        for i, linea in enumerate(instrucciones):
            color = (180, 220, 255) if i == 0 else (180, 200, 180)
            fuente = self.font_chica
            texto = fuente.render(linea, True, color)
            self.screen.blit(texto, (740, y_pos))
            y_pos += 25
    
    def ejecutar(self):
        """Bucle principal del programa"""
        print("\n▶️  Iniciando detección...")
        print("   Haz expresiones faciales frente a la cámara")
        
        pausado = False
        
        while self.ejecutando:
            if not pausado:
                # 1. LEER CÁMARA
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error leyendo cámara")
                    break
                
                # Voltear horizontalmente (como espejo)
                frame = cv2.flip(frame, 1)
                
                # 2. DETECTAR ROSTROS
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rostros = self.face_cascade.detectMultiScale(
                    gris, 
                    scaleFactor=1.1, 
                    minNeighbors=6, 
                    minSize=(100, 100)
                )
                
                self.rostro_detectado = len(rostros) > 0
                
                # 3. PROCESAR ROSTRO
                if len(rostros) > 0:
                    # Tomar el rostro más grande
                    x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
                    
                    # Dibujar rectángulo alrededor del rostro
                    color_rect = self.colores.get(self.expresion_actual, (0, 255, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color_rect, 3)
                    
                    # Recortar rostro para análisis
                    rostro_gris = gris[y:y+h, x:x+w]
                    
                    # 4. ANALIZAR EXPRESIÓN
                    expresion, confianza = self.analizar_expresion(rostro_gris)
                    
                    # Actualizar con filtro temporal (evita cambios bruscos)
                    tiempo_actual = time.time()
                    if tiempo_actual - self.ultimo_cambio > 0.4:  # 400ms entre cambios
                        self.expresion_actual = expresion
                        self.confianza = confianza
                        self.ultimo_cambio = tiempo_actual
                    
                    # Mostrar expresión en video
                    cv2.putText(
                        frame, 
                        f"{self.expresion_actual.upper()}", 
                        (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        color_rect, 
                        2
                    )
                else:
                    self.expresion_actual = "neutral"
                    self.confianza = 0.0
                
                # 5. CONVERTIR PARA PYGAME
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rotado = np.rot90(frame_rgb)
                frame_pygame = pygame.surfarray.make_surface(frame_rotado)
            
            # 6. DIBUJAR INTERFAZ
            self.dibujar_interfaz(frame_pygame)
            
            if pausado:
                # Texto de pausa
                pausa_text = self.font_grande.render("⏸️  PAUSADO", True, (255, 100, 100))
                pausa_rect = pausa_text.get_rect(center=(550, 300))
                self.screen.blit(pausa_text, pausa_rect)
            
            # 7. MANEJAR EVENTOS
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.ejecutando = False
                elif evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_ESCAPE:
                        self.ejecutando = False
                        print("\n🛑 Programa finalizado")
                    elif evento.key == pygame.K_SPACE:
                        pausado = not pausado
                        print(f"⏸️  Pausa: {'ACTIVADA' if pausado else 'DESACTIVADA'}")
            
            # 8. ACTUALIZAR PANTALLA
            pygame.display.flip()
            pygame.time.Clock().tick(30)  # 30 FPS
        
        # 9. LIMPIAR RECURSOS
        self.finalizar()
    
    def finalizar(self):
        """Libera todos los recursos"""
        print("\n🧹 Limpiando recursos...")
        self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
        print("✅ Programa finalizado correctamente")

# ============================================
# PUNTO DE ENTRADA
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("🎭 DETECTOR DE EXPRESIONES FACIALES")
    print("=" * 50)
    
    try:
        detector = DetectorExpresiones()
        detector.ejecutar()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)[:100]}...")
        print("\n💡 SOLUCIONES RÁPIDAS:")
        print("1. No necesitas imágenes - el programa crea automáticas")
        print("2. Si tienes cámara externa, prueba desactivarla")
        print("3. Cierra otras apps que usen la cámara")
        print("4. Ejecuta como administrador si hay problemas de permisos")