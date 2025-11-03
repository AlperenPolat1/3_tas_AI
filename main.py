import pygame
import sys
from functools import lru_cache

# -------------------------------------------------------------
# Üç Taş – Human vs AI (Minimax + Alpha-Beta)
# -------------------------------------------------------------
# Kurallar :
# - 3x3 noktadan oluşan tahtada her oyuncunun 3 taşı vardır.
# - Aşama 1 (Yerleştirme): Oyuncular sırayla boş noktalara taş koyar (toplam 6 hamle).
# - Aşama 2 (Hareket): Her hamlede kendi taşlarından birini, KOMŞU boş bir noktaya sürükler/taşır.
# - Amaç: Yatay veya dikey (3-in-a-row) yapmak.
# - İlk üçlü yapan kazanır.
# -------------------------------------------------------------

WIDTH, HEIGHT = 600, 600
MARGIN = 60
BG_COLOR = (245, 245, 245)
LINE_COLOR = (40, 40, 40)
HUMAN_COLOR = (22, 155, 98)
AI_COLOR = (180, 35, 35)
SEL_COLOR = (30, 120, 200)
FONT_COLOR = (20, 20, 20)

# --- AI mod seçimi (2 mod) ---
# 'imkansız': en iyi hamleleri oynar
# 'kolay': kasıtlı olarak kaybetmeye çalışır
AI_MODE = 'imkansız'


# Tahta 3x3 noktalar, index: 0..8 e kadar
# 0 1 2
# 3 4 5
# 6 7 8

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # satırlar
    (0, 3, 6), (1, 4, 7), (2, 5, 8)   # sütunlar
]

# Komşuluk (hareket aşaması). Üç Taş için tipik bağlantılar: yatay/dikey + merkezden köşelere.
# Köşeler: ortadaki kenar ve merkeze bağlı
# Kenarlar: komşu köşe ve merkeze bağlı
# Merkez: tüm kenarlar ve köşelerle bağlantılı değil; yaygın varyant merkez-köşe ve merkez-kenar bağlantılarını kabul eder.
# Burada 4-yön ve merkez-köşe bağlantıları verildi.
ADJ = {
    #  komşuluk durumları
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7]
}

EMPTY, HUMAN, AI = 0, 1, 2

# Pygame kurulum
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Üç Taş – Minimax AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 22)


def grid_pos(idx):
    r, c = divmod(idx, 3)
    x = MARGIN + c * (WIDTH - 2 * MARGIN) // 2
    y = MARGIN + r * (HEIGHT - 2 * MARGIN) // 2
    return x, y


def draw_board(board, selected=None, info_text=""):
    screen.fill(BG_COLOR)

    # Izgara çizgileri
    # Dikey
    pygame.draw.line(screen, LINE_COLOR, (WIDTH//2, MARGIN), (WIDTH//2, HEIGHT - MARGIN), 4)
    pygame.draw.line(screen, LINE_COLOR, (MARGIN + (WIDTH - 2*MARGIN)//2, MARGIN), (MARGIN + (WIDTH - 2*MARGIN)//2, HEIGHT - MARGIN), 4)
    # Yatay
    pygame.draw.line(screen, LINE_COLOR, (MARGIN, HEIGHT//2), (WIDTH - MARGIN, HEIGHT//2), 4)
    pygame.draw.line(screen, LINE_COLOR, (MARGIN, MARGIN + (HEIGHT - 2*MARGIN)//2), (WIDTH - MARGIN, MARGIN + (HEIGHT - 2*MARGIN)//2), 4)

    # Noktalar
    for i in range(9):
        x, y = grid_pos(i)
        pygame.draw.circle(screen, LINE_COLOR, (x, y), 8)

    # Taşlar
    for i, cell in enumerate(board):
        if cell == EMPTY:
            continue
        x, y = grid_pos(i)
        color = HUMAN_COLOR if cell == HUMAN else AI_COLOR
        radius = 28
        pygame.draw.circle(screen, color, (x, y), radius)
        if selected == i:
            pygame.draw.circle(screen, SEL_COLOR, (x, y), radius + 4, 4)

    # Bilgi metni
    info_surface = font.render(info_text, True, FONT_COLOR)
    screen.blit(info_surface, (MARGIN, HEIGHT - MARGIN + 10 if HEIGHT - MARGIN + 10 < HEIGHT else HEIGHT - 30))

    pygame.display.flip()


def is_win(board, player):
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] == player:
            return True
    return False


def count_pieces(board, player):
    return sum(1 for x in board if x == player)


def phase(board):
    # Toplam yerleştirilen taş 6'dan küçükse yerleştirme aşaması devam etmeli o yüzden retrun place
    if board.count(HUMAN) + board.count(AI) < 6:
        return "place"
    return "move"


def legal_moves(board, player):
    ph = phase(board)
    moves = []
    if ph == "place":
        for i in range(9):
            if board[i] == EMPTY:
                moves.append((None, i))  # yerleştirme açaması olduğu için None
    else:
        for i in range(9):
            if board[i] == player:
                for j in ADJ[i]:
                    if board[j] == EMPTY:
                        moves.append((i, j))     # oynama aşaması olduğu için None değil
    return moves


def apply_move(board, move, player):
    fr, to = move
    newb = list(board)
    if fr is None:
        newb[to] = player
    else:
        newb[fr] = EMPTY
        newb[to] = player
    return tuple(newb)

@lru_cache(maxsize=None)
def minimax(board, player, depth, alpha, beta):
    # AI maksimize eder; AI kazanımı pozitif
    if is_win(board, AI):
        return 10 - depth, None
    if is_win(board, HUMAN):
        return depth - 10, None
    if is_win(board, AI):
        return depth - 10, None

    # Berabere/hamle yok: 0
    moves = legal_moves(board, player)
    if not moves or depth >= 16:  # loopa düşmesib diye depth sınırlı
        return evaluate(board), None

    if player == AI:
        best_val, best_move = -float('inf'), None
        for mv in moves:
            newb = apply_move(board, mv, AI)
            val, _ = minimax(newb, HUMAN, depth+1, alpha, beta)
            if val > best_val:
                best_val, best_move = val, mv
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val, best_move = float('inf'), None
        for mv in moves:
            newb = apply_move(board, mv, HUMAN)
            val, _ = minimax(newb, AI, depth+1, alpha, beta)
            if val < best_val:
                best_val, best_move = val, mv
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move

def evaluate(board):
    # Basit heuristik: AI'nin 2'li doğruları +, HUMAN'ın 2'lileri -
    score = 0
    for a, b, c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        if line.count(AI) == 3:
            return 100
        if line.count(HUMAN) == 3:
            return -100
        if line.count(AI) == 2 and line.count(EMPTY) == 1:
            score += 3
        if line.count(HUMAN) == 2 and line.count(EMPTY) == 1:
            score -= 3
    # ödüllenidrme
    if board[4] == AI:
        score += 1
    if board[4] == HUMAN:
        score -= 1
    return score

def ai_move(board):
    moves = legal_moves(board, AI)
    if not moves:
        return None
       # 2 mod yapmak zorunda kaldım çünkü ai yenilemez bi hal aldı :'(
    if AI_MODE == 'imkansız':
        # EĞER DOĞRUDAN KAZANDIRAN HAMLE VARSA ÖNCE ONU YAPTIRDIM
        for mv in moves:
            if is_win(apply_move(tuple(board), mv, AI), AI):
                return mv
        # Yoksa minimax ile en iyi hamleyi buluyor
        _, mv = minimax(tuple(board), AI, 0, -float('inf'), float('inf'))
        return mv

    elif AI_MODE == 'kolay':
        # Minimax değerine göre AI için EN KÖTÜ hamleyi seçtirdim.
        worst_val, worst_move = float('inf'), None
        for mv in moves:
            newb = apply_move(tuple(board), mv, AI)
            val, _ = minimax(newb, HUMAN, 1, -float('inf'), float('inf'))
            if val < worst_val:
                worst_val, worst_move = val, mv
        return worst_move

    else:
        # seçilmezse
        _, mv = minimax(tuple(board), AI, 0, -float('inf'), float('inf'))
        return mv
    # Yoksa minimax ile en iyi hamleyi bul
    _, mv = minimax(tuple(board), AI, 0, -float('inf'), float('inf'))
    return mv

def human_click_to_index(pos):     # tam olarak noktaya basmak zorunda değiliz artık
    x, y = pos
    best_i, best_d2 = None, float('inf')
    for i in range(9):
        gx, gy = grid_pos(i)
        d2 = (x - gx) ** 2 + (y - gy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    # yalnızca noktaya yeterince yakınsa kabul et
    if best_d2 <= 40 ** 2:
        return best_i
    return None

def main():
    global AI_MODE
    board = [EMPTY] * 9
    turn = HUMAN # biz başlayalım                              # aklımdan sırayla başlayan
    selected = None                   #   bir oyun yapmak da geçti ama böylesi daha iyi zaten kazanmıyorum
    running = True

    while running:
        ph = "Yerleştirme" if phase(board) == "place" else "Hareket"
        info = f" {ph} | Mod: {AI_MODE} (1:imkansız,2:kolay,T:geçiş)"
        if is_win(board, HUMAN):
            info = "Kazanan: İnsan"
        elif is_win(board, AI):
            info = "Kazanan: Yapay Zeka"

        draw_board(board, selected, info)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if is_win(board, HUMAN) or is_win(board, AI):
                continue

            # AI modu kısayolları: 1=imkansız, 2=kolay, T=iki mod arası geçişler için
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    AI_MODE = 'imkansız'
                elif event.key == pygame.K_2:
                    AI_MODE = 'kolay'
                elif event.key == pygame.K_t:
                    AI_MODE = 'kolay' if AI_MODE == 'imkansız' else 'imkansız'

            if turn == HUMAN and event.type == pygame.MOUSEBUTTONDOWN:
                idx = human_click_to_index(pygame.mouse.get_pos())
                if idx is None:
                    continue
                if phase(board) == "place": #yerleştirirken
                    if board[idx] == EMPTY:
                        board[idx] = HUMAN
                        turn = AI
                else:  # hareket kısmı
                    if selected is None:
                        if board[idx] == HUMAN:
                            selected = idx
                    else:
                        # seçili taşı komşu boş noktaya taşı
                        if board[idx] == EMPTY and idx in ADJ[selected]:
                            board[selected] = EMPTY
                            board[idx] = HUMAN
                            selected = None
                            turn = AI
                        elif board[idx] == HUMAN:
                            # başka taşı seçme
                            selected = idx

        if running and not (is_win(board, HUMAN) or is_win(board, AI)) and turn == AI:
            # AI hamlesi
            mv = ai_move(board)
            if mv is None:
                # eğer hamle yoksa (böyle bi durum olmaz çünkü hamle hep var)
                turn = HUMAN
            else:
                fr, to = mv
                if fr is None:
                    board[to] = AI
                else:
                    board[fr] = EMPTY
                    board[to] = AI
                turn = HUMAN

        clock.tick(60)
    pygame.quit()
    sys.exit()
if __name__ == "__main__":
    main()
