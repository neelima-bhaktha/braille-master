from http.server import HTTPServer, BaseHTTPRequestHandler
from imutils.perspective import four_point_transform as FPT
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from imutils import contours
import numpy as np
import imutils
import cv2
import re
import json
import cgi
import io
import base64
import warnings
warnings.filterwarnings("ignore")


# ── Braille pipeline (your original code, wrapped in a class) ──────────────

def run_pipeline(image_bytes, iter=0, width=1500):

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if width:
        image = imutils.resize(image, width)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    ctrs = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(ctrs)

    paper = image.copy()
    gray2 = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    ctrs = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(ctrs)

    # get diameter
    boundingBoxes = [list(cv2.boundingRect(c)) for c in ctrs]
    c = Counter([i[2] for i in boundingBoxes])
    mode = c.most_common(1)[0][0]
    diam = mode if mode > 1 else c.most_common(2)[1][0]

    # get circles
    questionCtrs = []
    for c in ctrs:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if diam * 0.8 <= w <= diam * 1.2 and 0.8 <= ar <= 1.2:
            questionCtrs.append(c)

    # sort contours
    BB = [list(cv2.boundingRect(c)) for c in questionCtrs]
    tol = 0.7 * diam

    def sort_coord(i):
        S = sorted(BB, key=lambda x: x[i])
        s = [b[i] for b in S]
        m = s[0]
        for b in S:
            if m - tol < b[i] < m or m < b[i] < m + tol:
                b[i] = m
            elif b[i] > m + diam:
                for e in s[s.index(m):]:
                    if e > m + diam:
                        m = e
                        break
        return sorted(set(s))

    xs = sort_coord(0)
    ys = sort_coord(1)
    (questionCtrs, BB) = zip(*sorted(zip(questionCtrs, BB), key=lambda b: b[1][1] * len(image) + b[1][0]))
    boundingBoxes = list(BB)

    # draw contours on paper
    for q in range(len(questionCtrs)):
        cv2.drawContours(paper, questionCtrs[q], -1, (0, 255, 0), 3)

    # get spacing
    def spacing(x):
        space = []
        coor = [b[x] for b in boundingBoxes]
        for i in range(len(coor) - 1):
            c = coor[i + 1] - coor[i]
            if c > diam // 2:
                space.append(c)
        return sorted(list(set(space)))

    spacingX = spacing(0)
    spacingY = spacing(1)

    d1 = spacingX[0]
    d2 = 0
    d3 = 0
    for x in spacingX:
        if d2 == 0 and x > d1 * 1.3:
            d2 = x
        if d2 > 0 and x > d2 * 1.3:
            d3 = x
            break

    linesV = []
    prev = 0
    linesV.append(min(xs) - (d2 - diam) / 2)

    for i in range(1, len(xs)):
        diff = xs[i] - xs[i - 1]
        if i == 1 and d2 * 0.9 < diff:
            linesV.append(min(xs) - d2 - diam / 2)
            prev = 1
        if d1 * 0.8 < diff < d1 * 1.2:
            linesV.append(xs[i - 1] + diam + (d1 - diam) / 2)
            prev = 1
        elif d2 * 0.8 < diff < d2 * 1.1:
            linesV.append(xs[i - 1] + diam + (d2 - diam) / 2)
            prev = 0
        elif d3 > 0 and d3 * 0.9 < diff < d3 * 1.1:
            if prev == 1:
                linesV.append(xs[i - 1] + diam + (d2 - diam) / 2)
                linesV.append(xs[i - 1] + d2 + diam + (d1 - diam) / 2)
            else:
                linesV.append(xs[i - 1] + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + diam + (d2 - diam) / 2)
        elif d3 > 0 and d3 * 1.1 < diff:
            if prev == 1:
                linesV.append(xs[i - 1] + diam + (d2 - diam) / 2)
                linesV.append(xs[i - 1] + d2 + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d3 + diam + (d2 - diam) / 2)
                prev = 0
            else:
                linesV.append(xs[i - 1] + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + diam + (d2 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + d2 + diam + (d1 - diam) / 2)
                linesV.append(xs[i - 1] + d1 + d3 + diam + (d2 - diam) / 2)
                prev = 1

    linesV.append(max(xs) + diam * 1.5)
    if len(linesV) % 2 == 0:
        linesV.append(max(xs) + d2 + diam)

    # get letters
    Bxs = list(boundingBoxes)
    Bxs.append((100000, 0))

    dots = [[]]
    for y in sorted(list(set(spacingY))):
        if y > 1.3 * diam:
            minYD = y * 1.5
            break

    for b in range(len(Bxs) - 1):
        if Bxs[b][0] < Bxs[b + 1][0]:
            dots[-1].append(Bxs[b][0])
        else:
            if abs(Bxs[b + 1][1] - Bxs[b][1]) < minYD:
                dots[-1].append(Bxs[b][0])
                dots.append([])
            else:
                dots[-1].append(Bxs[b][0])
                dots.append([])
                if len(dots) % 3 == 0 and not dots[-1]:
                    dots.append([])

    letters = []
    for r in range(len(dots)):
        if not dots[r]:
            letters.append([0 for _ in range(len(linesV) - 1)])
            continue
        else:
            letters.append([])
            c = 0
            i = 0
            while i < len(linesV) - 1:
                if c < len(dots[r]):
                    if linesV[i] < dots[r][c] < linesV[i + 1]:
                        letters[-1].append(1)
                        c += 1
                    else:
                        letters[-1].append(0)
                else:
                    letters[-1].append(0)
                i += 1

    # translate
    alpha = {
        'a': '1', 'b': '13', 'c': '12', 'd': '124', 'e': '14', 'f': '123',
        'g': '1234', 'h': '134', 'i': '23', 'j': '234', 'k': '15',
        'l': '135', 'm': '125', 'n': '1245', 'o': '145', 'p': '1235',
        'q': '12345', 'r': '1345', 's': '235', 't': '2345', 'u': '156',
        'v': '1356', 'w': '2346', 'x': '1256', 'y': '12456', 'z': '1456',
        '#': '2456', '^': '26', ',': '3', '.': '346', '"': '356',
        ':': '34', "'": '5'
    }
    nums = {'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
            'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '0'}
    braille = {v: k for k, v in alpha.items()}

    letters_arr = np.array([np.array(l) for l in letters])
    ans = ''
    for r in range(0, len(letters_arr), 3):
        for c in range(0, len(letters_arr[0]), 2):
            f = letters_arr[r:r + 3, c:c + 2].flatten()
            f = ''.join([str(i + 1) for i, d in enumerate(f) if d == 1])
            if f == '6':
                f = '26'
            if not f:
                if ans and ans[-1] != ' ':
                    ans += ' '
            elif f in braille:
                ans += braille[f]
            else:
                ans += '?'
        if ans and ans[-1] != ' ':
            ans += ' '

    def replace_nums(m):
        return nums.get(m.group('key'), m.group(0))
    ans = re.sub('#(?P<key>[a-zA-Z])', replace_nums, ans)

    def capitalize(m):
        return m.group(0).upper()[1]
    ans = re.sub(r'\^(?P<key>[a-zA-Z])', capitalize, ans)

    # encode processed image to base64
    _, buf = cv2.imencode('.jpg', paper)
    img_b64 = base64.b64encode(buf).decode('utf-8')

    return {
        "text": ans.strip(),
        "dot_count": len(questionCtrs),
        "processed_image": f"data:image/jpeg;base64,{img_b64}"
    }


# ── HTTP Handler ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  {args[0]} {args[1]}")

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            with open('braille_frontend.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/translate':
            ctype, pdict = cgi.parse_header(self.headers.get('Content-Type'))
            pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
            pdict['CONTENT-LENGTH'] = int(self.headers.get('Content-Length'))
            fields = cgi.parse_multipart(self.rfile, pdict)

            image_bytes = fields.get('image')[0]
            width = int(fields.get('width', [1500])[0])
            iter_ = int(fields.get('iter', [0])[0])

            try:
                result = run_pipeline(image_bytes, iter=iter_, width=width)
                body = json.dumps(result).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    port = 8000
    print(f"\n  Braille Reader running at http://localhost:{port}\n")
    HTTPServer(('', port), Handler).serve_forever()
