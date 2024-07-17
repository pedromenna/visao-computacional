import cv2
import numpy as np
import time

ARQUIVO_VIDEO = 'estrada.mp4'
ARQUIVO_CFG = 'yolov3.cfg'
ARQUIVO_WEIGHTS = 'yolov3.weights'
ARQUIVO_CLASSES = 'coco.names'

def carregar_classes(arquivo):
    with open(arquivo, 'r') as f:
        return [line.strip() for line in f.readlines()]

def desenhar_caixas(img, caixas, confiancas, indices, classes, classes_ids, velocidades):
    for i in indices.flatten():
        caixa = caixas[i]
        (x, y, w, h) = (caixa[0], caixa[1], caixa[2], caixa[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        texto = f"{classes[classes_ids[i]]}" 
        cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if i in velocidades and velocidades[i] >= 35: 
            velocidade_texto = f"Velocidade: {velocidades[i]:.2f} km/h"
            cv2.putText(img, velocidade_texto, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def calcular_velocidade(posicao_antiga, posicao_nova, tempo, escala_pixel_para_metros):
    distancia = np.linalg.norm(np.array(posicao_nova) - np.array(posicao_antiga)) * escala_pixel_para_metros
    velocidade_m_s = distancia / tempo
    velocidade_kmh = velocidade_m_s * 3.6
    return velocidade_kmh

def main():
    classes = carregar_classes(ARQUIVO_CLASSES)
    net = cv2.dnn.readNet(ARQUIVO_WEIGHTS, ARQUIVO_CFG)
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    
    if not captura.isOpened():
        print(f"Erro ao abrir o vídeo {ARQUIVO_VIDEO}")
        return

    fps = captura.get(cv2.CAP_PROP_FPS)
    intervalo_tempo = 1.0 / fps
    escala_pixel_para_metros = 0.05

    posicoes_anteriores = {}
    tempos_anteriores = {}
    velocidades = {}

    frame_counter = 0
    frame_skip = int(fps * 4)

    while True:
        ret, frame = captura.read()
        if not ret:
            print("Erro ao ler o frame do vídeo ou fim do vídeo.")
            break

        if frame_counter % frame_skip == 0:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            saidas = net.forward(net.getUnconnectedOutLayersNames())

            caixas = []
            confiancas = []
            classes_ids = []

            for saida in saidas:
                for deteccao in saida:
                    scores = deteccao[5:]
                    class_id = np.argmax(scores)
                    confianca = scores[class_id]
                    if confianca > 0.5:
                        caixa_centro_x = int(deteccao[0] * frame.shape[1])
                        caixa_centro_y = int(deteccao[1] * frame.shape[0])
                        largura = int(deteccao[2] * frame.shape[1])
                        altura = int(deteccao[3] * frame.shape[0])
                        x = int(caixa_centro_x - largura / 2)
                        y = int(caixa_centro_y - altura / 2)
                        caixas.append([x, y, largura, altura])
                        confiancas.append(float(confianca))
                        classes_ids.append(class_id)

                        posicao_atual = (caixa_centro_x, caixa_centro_y)
                        if class_id in posicoes_anteriores:
                            tempo_decorrido = (frame_counter - tempos_anteriores[class_id]) * intervalo_tempo
                            velocidade = calcular_velocidade(posicoes_anteriores[class_id], posicao_atual, tempo_decorrido, escala_pixel_para_metros)
                            if velocidade < 300:
                                velocidades[class_id] = velocidade
                        posicoes_anteriores[class_id] = posicao_atual
                        tempos_anteriores[class_id] = frame_counter

            indices = cv2.dnn.NMSBoxes(caixas, confiancas, 0.5, 0.4)

            if len(indices) > 0:
                desenhar_caixas(frame, caixas, confiancas, indices, classes, classes_ids, velocidades)
                cv2.imshow("Detecção de Veículos", frame)
                if cv2.waitKey(3000) & 0xFF == ord('q'): 
                    break
        else:
            # Exibir o frame normal
            cv2.imshow("Detecção de Veículos", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_counter += 1

    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
