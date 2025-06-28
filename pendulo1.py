import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statistics

# função que calcula fator de conversão (cm/pixel) usando os contornos da bolinha
def calcular_fator(contornos, diametro_real_cm=2.5):
    # Filtra áreas maiores que 50 para evitar ruídos pequenos
    areas_validas = [cv2.contourArea(c) for c in contornos if cv2.contourArea(c) > 50]
    
    # Se não encontrar áreas válidas, retorna um valor padrão
    if not areas_validas:
        return 0.00375

    # Calcula o diâmetro em pixels para cada área válida (área = π * r² => d = 2r)
    diametros_px = 2 * np.sqrt(np.array(areas_validas) / np.pi)
    
    # Fator é a razão entre o diâmetro real (cm) e o diâmetro médio em pixels,
    # ou seja, quantos centímetros correspondem a 1 pixel na imagem
    return diametro_real_cm / np.mean(diametros_px)

# função que gera gráficos da posição X e Y em função do tempo e calcula posição ideal do pêndulo
def grafico_posicao_tempo(centros, comprimento, fator_cm_px, x_central, fps):
    pos_x = []
    pos_y = []
    tempos = []

    # Converte as posições de pixel para cm e calcula o tempo para cada frame
    for i, ponto in enumerate(centros):
        pos_x.append((x_central - ponto[0]) * fator_cm_px)  # Posição X relativa ao centro (cm)
        pos_y.append(ponto[1] * fator_cm_px)                # Posição Y (cm)
        tempos.append(i / fps)                              # Tempo em segundos

    # Descarta os primeiros 10 pontos (pontos sendo detectados que não eram a bolinha)
    pos_x = pos_x[10:]
    pos_y = pos_y[10:]
    tempos = tempos[10:]

    # Remove valores muito distantes da média para X, considerando 3 desvios padrões (filtragem estatística)
    media_x = statistics.mean(pos_x)
    desvio_x = statistics.stdev(pos_x)
    limite = 3
    filtrados = [(x, y, t) for x, y, t in zip(pos_x, pos_y, tempos) if abs(x - media_x) < limite * desvio_x]

    # Atualiza listas com os valores filtrados
    pos_x = [x for x, y, t in filtrados]
    pos_y = [y for x, y, t in filtrados]
    tempos = [t for x, y, t in filtrados]

    # Salva gráficos de posição X e Y ao longo do tempo (útil para análise visual)
    plt.plot(tempos, pos_x, marker='o')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição X (cm)")
    plt.title("Posição X vs Tempo")
    plt.savefig('grafico_pos_x.png')
    plt.clf()

    plt.plot(tempos, pos_y, marker='o')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição Y (cm)")
    plt.title("Posição Y vs Tempo")
    plt.savefig('grafico_pos_y.png')
    plt.clf()

    # --- CÁLCULO DA OSCILAÇÃO IDEAL DO PÊNDULO ---
    # O ângulo inicial do pêndulo é calculado com base na posição inicial X convertida em radianos,
    # considerando a pequena oscilação (aproximação do seno para ângulos pequenos)
    x0 = (centros[0][0] - x_central) * fator_cm_px
    ang_inicial = np.arcsin(x0 * 0.01 / comprimento)  # converte para metros

    g = 9.81  # aceleração da gravidade (m/s²)
    omega = np.sqrt(g / comprimento)  # frequência angular do pêndulo simples (rad/s)

    # Calcula a posição ideal do pêndulo para cada tempo usando fórmula teórica:
    # x(t) = L * sin(θ₀ * cos(ω * t)), que vem da solução do pêndulo simples em pequenas oscilações
    pos_ideal_x = [comprimento * 100 * np.sin(ang_inicial * np.cos(omega * t)) for t in tempos]

    plt.plot(tempos, pos_ideal_x, linestyle='--', label='Ideal - X')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição X Ideal (cm)")
    plt.title("Oscilação ideal do pêndulo")
    plt.legend()
    plt.savefig('grafico_ideal_pendulo.png')
    plt.clf()

    return tempos, pos_x

# função que ajusta modelo de oscilador harmônico amortecido e imprime parâmetros físicos
def ajustaOHA(time, coordinates_X, L):
    time = np.array(time)
    coordinates_X = np.array(coordinates_X)

    def modelo_OHA(t, A, b, omega, phi):
        return A * np.exp(-b * t) * np.cos(omega * t + phi)

    A_guess = np.max(np.abs(coordinates_X))
    b_guess = 0.01
    omega_guess = np.sqrt(9.81 / L)
    phi_guess = 0

    try:
        params, _ = curve_fit(
            modelo_OHA, time, coordinates_X,
            p0=[A_guess, b_guess, omega_guess, phi_guess],
            bounds=([0, 0, 0, -2*np.pi], [np.inf, 1, 2*np.pi*10, 2*np.pi]),
            maxfev=10000)
    except RuntimeError:
        print("Ajuste OHA não convergiu, usando parâmetros iniciais.")
        params = [A_guess, b_guess, omega_guess, phi_guess]

    A, b, omega, phi = params
    Q = omega / (2 * b) if b != 0 else np.inf

    x_ajustado = modelo_OHA(time, A, b, omega, phi)
    plt.scatter(time, coordinates_X, label='Dados', color='blue', s=10)
    plt.plot(time, x_ajustado, label='Ajuste OHA', color='red')
    plt.xlabel("Tempo (s)")
    plt.ylabel("X (cm)")
    plt.title("Ajuste do OHA")
    plt.legend()
    plt.savefig('ajuste_oha.png')
    plt.clf()

    print(f"\n--- Parâmetros do OHA ---\nA: {A:.4f} cm\nb: {abs(b):.4f} s⁻¹\nomega: {omega:.4f} rad/s\nphi: {phi:.4f} rad\nQ: {abs(Q):.4f}")

    return A, b, omega, phi, Q


# função que gera tabela com tempos e posições X e Y para análise
def gerar_tabela_posicoes(centros, fator_cm_px, x_central, fps):
    tabela = []
    intervalo = int(fps * 2)  # seleciona pontos a cada 2 segundos para a tabela

    for i, ponto in enumerate(centros):
        if i % intervalo == 0:
            tempo = round(i / fps, 2)
            pos_x = round((ponto[0] - x_central) * fator_cm_px, 3)
            pos_y = round(ponto[1] * fator_cm_px, 3)
            tabela.append([tempo, pos_x, pos_y])

    cabecalhos = ["Tempo (s)", "Posição X (cm)", "Posição Y (cm)"]
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('tight')
    ax.axis('off')

    dados = np.array(tabela)
    tabela_plot = ax.table(cellText=dados, colLabels=cabecalhos, loc='center')
    tabela_plot.auto_set_font_size(False)
    tabela_plot.set_fontsize(12)
    tabela_plot.scale(1.2, 2)

    plt.savefig('tabela_posicoes.png')
    plt.close()

# --- PARÂMETROS DO EXPERIMENTO ---
video_entrada = 'video1.mp4'
video_saida = 'output_video1.mp4'
comprimento_fio = 0.52          # Comprimento do fio do pêndulo em metros
diametro_bolinha_cm = 2.5       # Diâmetro real da bolinha em centímetros

# Abre o vídeo para leitura
captura = cv2.VideoCapture(video_entrada)
if not captura.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Obtém propriedades do vídeo
largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = captura.get(cv2.CAP_PROP_FPS)

# Configura vídeo para salvar saída (com retângulos e marcação dos centros)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
saida = cv2.VideoWriter(video_saida, fourcc, fps_video, (largura, altura))

x_central = largura / 2  # eixo central X da imagem
centros_bolinha = []
lista_contornos = []

print(f"\nProcessando vídeo: {video_entrada}")
print(f"Resolução: {largura}x{altura}")
print(f"FPS: {fps_video:.2f}")

# Lê o primeiro frame para calcular o fator de conversão cm/pixel
ret, frame = captura.read()
if not ret:
    print("Erro ao ler o primeiro frame!")
    exit()

cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, mascara = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)  # Inverte preto e branco para facilitar detecção
kernel = np.ones((5, 5), np.uint8)
mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)       # Fecha pequenos buracos na máscara
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fator_cm_px = calcular_fator(contornos)
print(f"\nFator de conversão cm/pixel: {fator_cm_px:.6f}")

# Calcula área aproximada da bolinha em pixels para filtrar contornos
diametro_px = diametro_bolinha_cm / fator_cm_px
area_bolinha_px = np.pi * (diametro_px / 2) ** 2
area_minima = area_bolinha_px * 0.7
area_maxima = area_bolinha_px * 1.3

# Volta para o início do vídeo para processar todos os frames
captura.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Processa frame a frame para detectar a bolinha e marcar posição
while True:
    ret, frame = captura.read()
    if not ret:
        break

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mascara = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY_INV)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos por área para encontrar a bolinha
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area_minima < area < area_maxima:
            x, y, w, h = cv2.boundingRect(contorno)
            centro_x = x + w // 2
            centro_y = y + h // 2

            centros_bolinha.append([centro_x, centro_y])
            lista_contornos.append(contorno)

            # Desenha retângulo e centro no frame para visualização
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (centro_x, centro_y), 5, (255, 0, 0), -1)

    saida.write(frame)

# Libera arquivos de vídeo
captura.release()
saida.release()

print(f"\nVídeo processado! Salvo como '{video_saida}'")
print(f"Total de detecções: {len(centros_bolinha)}")

# Caso tenha detectado pontos suficientes, faz análise e ajustes
if len(centros_bolinha) > 0:
    # Adiciona o raio da bolinha para o comprimento do pêndulo, para mais precisão
    comprimento_total = comprimento_fio + diametro_bolinha_cm / 200  # converte cm para metros
    
    tempos, posicoes_x = grafico_posicao_tempo(centros_bolinha, comprimento_total, fator_cm_px, x_central, fps_video)
    
    if len(tempos) > 10:
        ajustaOHA(tempos, posicoes_x, comprimento_total)
    else:
        print("\nDados insuficientes para ajuste OHA")
    
    gerar_tabela_posicoes(centros_bolinha, fator_cm_px, x_central, fps_video)
else:
    print("\nNenhum contorno detectado.")

