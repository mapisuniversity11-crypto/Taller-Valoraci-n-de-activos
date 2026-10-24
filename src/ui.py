"""Elementos visuales de la interfaz (solo presentación, sin cálculos financieros)."""

# Paleta de la pantalla de inicio
_PLATA = "#c9ced6"
_VERDE = (74, 222, 128)
_ROJO = (248, 113, 113)


def _mezcla(c1: tuple, c2: tuple, t: float) -> str:
    """Color intermedio entre c1 (t=0) y c2 (t=1), en formato rgb()."""
    r, g, b = (round(a + (b - a) * t) for a, b in zip(c1, c2))
    return f"rgb({r},{g},{b})"


def _svg_arbol(niveles: int, inicio: float, paso: float) -> str:
    """SVG de un árbol binomial recombinante que se dibuja nivel por nivel.

    Geometría CRR: el nodo (k, j) tiene j subidas en k pasos. Su altura depende de
    (2j − k): subir y luego bajar lleva al mismo nodo que bajar y luego subir.
    """
    ancho, alto = 1600, 900
    x0, x1 = 260, 1340
    dx = (x1 - x0) / niveles
    dy = 560 / niveles

    def pos(k: int, j: int) -> tuple[float, float]:
        return x0 + k * dx, alto / 2 - (2 * j - k) * dy / 2

    ramas, nodos = [], []
    for k in range(niveles):
        demora = inicio + k * paso
        for j in range(k + 1):
            xa, ya = pos(k, j)
            for j2 in (j, j + 1):   # baja (j) y sube (j + 1)
                xb, yb = pos(k + 1, j2)
                ramas.append(
                    f'<path class="qa-rama" d="M{xa:.1f},{ya:.1f} L{xb:.1f},{yb:.1f}" pathLength="1" '
                    f'style="animation-delay:{demora:.2f}s"/>'
                )
    for k in range(niveles + 1):
        # cada nodo se enciende cuando llega la rama que lo alcanza
        demora = inicio - 0.15 if k == 0 else inicio + (k - 1) * paso + paso * 0.85
        for j in range(k + 1):
            x, y = pos(k, j)
            if k == niveles:   # nodos finales: verde arriba (precio alto), rojo abajo (precio bajo)
                color, radio = _mezcla(_ROJO, _VERDE, j / niveles), 7
            else:
                color, radio = _PLATA, (8 if k == 0 else 5)
            nodos.append(
                f'<circle class="qa-nodo" cx="{x:.1f}" cy="{y:.1f}" r="{radio}" fill="{color}" '
                f'style="animation-delay:{max(demora, 0):.2f}s; color:{color}"/>'
            )
    return (
        f'<svg class="qa-arbol" viewBox="0 0 {ancho} {alto}" preserveAspectRatio="xMidYMid meet" '
        f'aria-hidden="true"><g class="qa-grupo">{"".join(ramas)}{"".join(nodos)}</g></svg>'
    )


def html_intro(titulo: str = "Quantia",
               subtitulo: str = "Valoración de opciones y riesgo de portafolio",
               duracion: float = 6.0,
               niveles: int = 6,
               fuente: str = "Michroma") -> str:
    """Pantalla de bienvenida: un árbol binomial crece en la oscuridad y luego surge el título.

    Solo HTML + SVG + CSS (sin JavaScript). La capa cubre toda la ventana y se
    desvanece sola al final de `duracion` segundos.
    """
    paso = 0.32                                  # segundos entre niveles del árbol
    inicio_arbol = 0.3
    fin_arbol = inicio_arbol + niveles * paso + 0.3
    inicio_titulo = fin_arbol + 0.1
    salida = max(duracion - 0.9, inicio_titulo + 1.5)
    familia = fuente.replace(" ", "+")
    mayusculas = fuente == "Michroma"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={familia}&family=Montserrat:wght@300&display=swap');

.qa-intro {{
    position: fixed; inset: 0; z-index: 1000000; overflow: hidden;
    background: radial-gradient(ellipse at 45% 50%, #0d1016 0%, #000 72%);
    animation: qa-salida 0.9s ease-in {salida:.2f}s forwards;
}}
.qa-arbol {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.qa-grupo {{ animation: qa-atenua 1.2s ease {fin_arbol:.2f}s forwards; }}
.qa-rama {{
    fill: none; stroke: {_PLATA}; stroke-width: 1.6; stroke-linecap: round; opacity: .75;
    stroke-dasharray: 1; stroke-dashoffset: 1;
    animation: qa-traza {paso * 1.05:.2f}s ease-out forwards;
}}
.qa-nodo {{
    opacity: 0; transform-box: fill-box; transform-origin: center;
    filter: drop-shadow(0 0 6px currentColor);
    animation: qa-enciende .45s cubic-bezier(.2, 1.6, .4, 1) forwards;
}}
.qa-centro {{
    position: absolute; inset: 0; display: flex; flex-direction: column;
    align-items: center; justify-content: center; pointer-events: none;
}}
.qa-titulo {{
    font-family: '{fuente}', 'Segoe UI', sans-serif !important; font-weight: 400;
    font-size: clamp(3rem, 10vw, 8rem); line-height: 1.05;
    letter-spacing: {"0.08em" if mayusculas else "0.01em"};
    text-transform: {"uppercase" if mayusculas else "none"};
    background: linear-gradient(110deg,
        #5c626b 0%, #c9ced6 18%, #ffffff 26%, #8d939c 38%,
        #e6e9ee 52%, #6e747d 66%, #f4f6f9 80%, #7d838c 100%);
    background-size: 250% 100%;
    -webkit-background-clip: text !important; background-clip: text !important;
    color: transparent !important; -webkit-text-fill-color: transparent;
    opacity: 0;
    animation:
        qa-surge 1.4s cubic-bezier(.16, 1, .3, 1) {inicio_titulo:.2f}s forwards,
        qa-brillo 3s ease-in-out {inicio_titulo + 0.4:.2f}s infinite;
}}
.qa-linea {{
    height: 1px; width: 0; margin: 1.2rem 0 .9rem;
    background: linear-gradient(90deg, transparent, {_PLATA}, transparent);
    animation: qa-linea 1s ease-out {inicio_titulo + 0.6:.2f}s forwards;
}}
.qa-sub {{
    font-family: 'Montserrat', 'Segoe UI', sans-serif !important; font-weight: 300;
    font-size: clamp(.75rem, 1.5vw, 1rem); letter-spacing: .35em; text-transform: uppercase;
    color: #9aa1ab !important; opacity: 0; text-align: center; padding: 0 1rem;
    animation: qa-aparece .9s ease-out {inicio_titulo + 0.9:.2f}s forwards;
}}
@keyframes qa-traza    {{ to {{ stroke-dashoffset: 0; }} }}
@keyframes qa-enciende {{ 0% {{ opacity: 0; transform: scale(0); }} 100% {{ opacity: 1; transform: scale(1); }} }}
@keyframes qa-atenua   {{ to {{ opacity: .16; }} }}
@keyframes qa-surge {{
    0%   {{ opacity: 0; transform: scale(.6); filter: blur(12px); }}
    100% {{ opacity: 1; transform: scale(1); filter: blur(0) drop-shadow(0 0 20px rgba(200,215,235,.22)); }}
}}
@keyframes qa-brillo  {{ 0%, 100% {{ background-position: 100% 0; }} 50% {{ background-position: 0% 0; }} }}
@keyframes qa-linea   {{ to {{ width: min(420px, 60vw); }} }}
@keyframes qa-aparece {{ to {{ opacity: 1; }} }}
@keyframes qa-salida  {{ to {{ opacity: 0; visibility: hidden; pointer-events: none; }} }}

@media (prefers-reduced-motion: reduce) {{
    .qa-rama, .qa-nodo, .qa-titulo, .qa-sub, .qa-linea, .qa-grupo {{ animation: none; opacity: 1;
        stroke-dashoffset: 0; width: auto; }}
}}
</style>
<div class="qa-intro">
    {_svg_arbol(niveles, inicio_arbol, paso)}
    <div class="qa-centro">
        <div class="qa-titulo">{titulo}</div>
        <div class="qa-linea"></div>
        <div class="qa-sub">{subtitulo}</div>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Portada de metal líquido dorado (WebGL)
# Shader adaptado del componente "ShaderMetal" (21st.dev): cintas de metal que
# reflejan una iluminación de estudio. Ver créditos en el README.
# ---------------------------------------------------------------------------
_VERT_METAL = """attribute vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}"""

_FRAG_METAL = """precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_count;
uniform float u_iridescence;
uniform float u_pull;
uniform float u_dark;
uniform vec2 u_mouse;
uniform vec2 u_light;
uniform vec3 u_f0;

const float TAU = 6.28318530718;
const float TILT = -0.32;

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Smooth 1D value noise, for the brushed grain that runs along each ribbon.
float lines(float x, float seed) {
  float i = floor(x);
  float f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(hash(vec2(i, seed)), hash(vec2(i + 1.0, seed)), f);
}

vec2 rotate(vec2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

// The spine of ribbon k: three swells at different speeds, so it never
// settles into a loop. Near the pointer it leans toward it.
float spine(float x, float k, float lane, float t, vec2 m) {
  float y = lane;
  y += 0.13 * sin(1.15 * x + 0.55 * t + 2.1 * k);
  y += 0.06 * sin(2.6 * x - 0.8 * t + 1.3 * k + 0.7);
  y += 0.025 * sin(5.3 * x + 1.3 * t + 4.0 * k);
  float near = exp(-(x - m.x) * (x - m.x) * 5.0);
  y += (m.y - y) * 0.22 * u_pull * near;
  return y;
}

// A studio to reflect: sky and floor meet at a crisp horizon, a big
// softbox overhead, two strip lights and the key light. Linear light.
vec3 studio(vec3 r, float t) {
  float a = 0.28 * sin(t * 0.11);
  r.xz = rotate(r.xz, a);
  float y = r.y;

  vec3 top = mix(vec3(0.82, 0.83, 0.86), vec3(0.025, 0.025, 0.028), u_dark);
  vec3 horizon = mix(vec3(1.0), vec3(0.16, 0.16, 0.17), u_dark);
  vec3 ground = mix(vec3(0.05, 0.05, 0.055), vec3(0.004), u_dark);
  vec3 sky = mix(horizon, top, smoothstep(0.0, 0.85, y));
  vec3 floor_ = mix(ground * 2.4, ground, smoothstep(0.0, -0.5, y));
  vec3 c = mix(floor_, sky, smoothstep(-0.025, 0.025, y));

  // Overhead softbox, feathered.
  float box = smoothstep(0.6, 0.45, abs(r.x)) * smoothstep(0.42, 0.55, y) *
              smoothstep(1.0, 0.85, y);
  c += box * mix(1.6, 2.2, u_dark);

  // Tall strip lights left and right: warm and cool, faintly.
  float stripL = smoothstep(0.1, 0.03, abs(r.x + 0.78)) * smoothstep(-0.35, -0.1, y) * smoothstep(0.8, 0.5, y);
  float stripR = smoothstep(0.08, 0.02, abs(r.x - 0.72)) * smoothstep(-0.3, -0.05, y) * smoothstep(0.75, 0.45, y);
  c += stripL * vec3(1.25, 1.1, 0.95) * mix(1.0, 1.5, u_dark);
  c += stripR * vec3(0.9, 1.05, 1.3) * mix(0.8, 1.3, u_dark);

  // A low bounce card, so the undersides aren't dead.
  float card = smoothstep(0.7, 0.2, abs(r.x)) * smoothstep(-0.95, -0.7, y) * smoothstep(-0.4, -0.65, y);
  c += card * mix(0.35, 0.18, u_dark);
  return c;
}

vec3 tonemap(vec3 x) {
  x = max(x, 0.0);
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  float px = 1.0 / u_resolution.y;
  vec2 p = rotate((frag - 0.5 * u_resolution) * px, TILT);
  vec2 m = rotate((u_mouse - 0.5) * u_resolution * px, TILT);
  float t = u_time;

  vec3 L = normalize(vec3(u_light, 0.85));
  float count = u_count;
  float spread = mix(0.0, 0.3, clamp((count - 1.0) / 4.0, 0.0, 1.0)) + 0.08 * step(1.5, count);
  float baseWidth = mix(0.13, 0.075, clamp((count - 1.0) / 4.0, 0.0, 1.0));

  // Premultiplied, composited back to front.
  vec3 col = vec3(0.0);
  float alpha = 0.0;

  for (int i = 0; i < 5; i++) {
    float k = float(i);
    if (k >= count) break;
    float f = count > 1.0 ? k / (count - 1.0) : 0.5;
    float lane = (f - 0.5) * spread;
    // Farther ribbons drift a touch slower: a hint of depth.
    float tk = t * mix(0.82, 1.0, f);

    float x = p.x;
    float e = 0.004;
    float c = spine(x, k, lane, tk, m);
    float slope = (spine(x + e, k, lane, tk, m) - spine(x - e, k, lane, tk, m)) / (2.0 * e);
    float stretch = inversesqrt(1.0 + slope * slope);
    float d = (p.y - c) * stretch;

    // Twist along the length; width breathes.
    float theta = 1.5 * x + 0.45 * tk + 1.7 * k + 0.55 * sin(0.7 * x - 0.3 * tk + k);
    float ct = cos(theta);
    float st = sin(theta);
    float hw = baseWidth * (0.82 + 0.22 * sin(0.9 * x + 0.35 * tk + 2.3 * k));
    // Seen edge-on a ribbon still has a little thickness.
    float proj = hw * max(abs(ct), 0.035);

    // Soft contact shadow on whatever is behind, before the ribbon lands.
    float sd = d + 0.045;
    float shadowW = proj + 0.05;
    float shadow = (1.0 - smoothstep(0.35, 1.0, abs(sd) / shadowW)) * mix(0.2, 0.32, u_dark);
    col *= 1.0 - shadow;
    alpha = alpha + shadow * (1.0 - alpha);

    float v = d / (ct >= 0.0 ? proj : -proj);
    float aa = 1.2 * px / proj;
    float cover = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, abs(v));
    if (cover <= 0.0) continue;

    // Ribbon frame: T along, N2 across in the screen plane, Z toward us.
    vec3 T = normalize(vec3(1.0, slope, 0.0));
    vec3 N2 = normalize(vec3(-slope, 1.0, 0.0));
    vec3 B = ct * N2 + st * vec3(0.0, 0.0, 1.0);
    vec3 n = -st * N2 + ct * vec3(0.0, 0.0, 1.0);

    // A shallow arc across the width, rolling over at the edges, plus
    // slow ripples running down the length and a brushed grain.
    float roll = v * 0.45 + v * v * v * 0.5;
    float ripple = sin(x * 13.0 - tk * 2.2 + k * 3.1) * 0.06 + sin(x * 31.0 + tk * 1.3 + k) * 0.015;
    float grain = lines(v * 34.0 + k * 17.0, k) - 0.5;
    n = normalize(n + B * (roll + grain * 0.015) + T * ripple);
    // Edge-on, what shows is the strip's polished edge. It faces us, tipped
    // up a little so it catches the sky rather than sitting on the horizon.
    float edge = smoothstep(0.14, 0.03, abs(ct));
    vec3 rim = B * (st >= 0.0 ? 1.0 : -1.0) + vec3(0.0, 0.2, 0.0);
    n = normalize(mix(n, normalize(rim), edge));
    float back = step(n.z, 0.0);
    n *= 1.0 - 2.0 * back;

    vec3 r = reflect(vec3(0.0, 0.0, -1.0), n);
    vec3 env = studio(r, t);

    float facing = clamp(n.z, 0.0, 1.0);
    vec3 F = u_f0 + (1.0 - u_f0) * pow(1.0 - facing, 5.0);

    // Thin film: thickness drifts along the ribbon, and the colour shifts
    // with the viewing angle, strongest on the turns.
    float film = 0.5 + 0.5 * sin(x * 1.6 + v * 0.9 + tk * 0.35 + k * 1.9);
    vec3 rainbow = 0.5 + 0.5 * cos(TAU * (film * 1.6 + facing * 1.1 + vec3(0.0, 0.33, 0.67)));
    // Keep the film's brightness level, so it tints the metal, never dims it.
    rainbow = 0.2 + 0.8 * rainbow;
    rainbow /= max(dot(rainbow, vec3(0.2126, 0.7152, 0.0722)), 0.2);
    float sheen = u_iridescence * (0.45 + 0.55 * (1.0 - facing));
    F *= mix(vec3(1.0), rainbow, sheen);

    float rl = max(dot(r, L), 0.0);
    float spec = pow(rl, 90.0) * 5.0 + pow(rl, 14.0) * 0.55;

    vec3 lit = env * F + spec * F;
    lit *= mix(1.0, 0.72, back) * (0.94 + grain * 0.12);
    lit *= mix(0.86, 1.0, f);
    vec3 rgb = linearToSrgb(tonemap(lit * 1.05));

    col = col * (1.0 - cover) + rgb * cover;
    alpha = alpha * (1.0 - cover) + cover;
  }

  // A hair of dither, so the long gradients never band. Faint shadows
  // need it on alpha too, or they step into contour lines.
  float dither = (hash(frag + fract(t) * 91.0) - 0.5) / 255.0;
  col += dither * alpha;
  alpha = clamp(alpha + dither * step(0.0005, alpha) * 1.5, 0.0, 1.0);
  gl_FragColor = vec4(clamp(col, 0.0, 1.0), alpha);
}"""

MARCA_PORTADA = "quantia-portada"


def css_portada_metal(duracion: float = 6.0) -> str:
    """CSS para la página de Streamlit: estira el iframe de la portada a pantalla
    completa y lo oculta al terminar. Si este CSS no aplicara, el iframe queda de
    1 px de alto (invisible), así que nunca tapa la app."""
    salida = max(duracion - 0.9, 1.0)
    # El "telón" es una capa negra de CSS puro: aparece en el mismo instante que este
    # bloque, antes de que el recuadro de la portada termine de cargar, y así la app
    # no se alcanza a ver por debajo. Se desvanece junto con la portada.
    return f"""<style>
.qp-telon {{
    position: fixed; inset: 0; z-index: 999999; background: #000;
    animation: qp-salida 0.9s ease-in {salida:.2f}s forwards;
}}
iframe[srcdoc*="{MARCA_PORTADA}"] {{
    position: fixed !important; inset: 0 !important;
    width: 100vw !important; height: 100vh !important;
    z-index: 1000000 !important; border: 0 !important; background: #000;
    animation: qp-salida 0.9s ease-in {salida:.2f}s forwards;
}}
@keyframes qp-salida {{ to {{ opacity: 0; visibility: hidden; pointer-events: none; }} }}
</style><div class="qp-telon"></div>"""


def html_portada_metal(titulo: str = "QUANTIA",
                       subtitulo: str = "Valoración de opciones y riesgo de portafolio",
                       duracion: float = 6.0,
                       cintas: int = 4,
                       color_metal: str = "#FFD38A",
                       iridiscencia: float = 0.12,
                       velocidad: float = 0.9) -> str:
    """Documento HTML (para st.components.v1.html) con cintas de oro líquido sobre
    negro y el título en blanco."""
    salida = max(duracion - 0.9, 1.0)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<!-- {MARCA_PORTADA} -->
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400&family=Inter:wght@300&display=swap" rel="stylesheet">
<style>
html, body {{ margin: 0; height: 100%; background: #000; overflow: hidden; }}
#lienzo {{ position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0;
          animation: qp-aparece 1.6s ease-out .1s forwards; }}
.velo {{ position: absolute; inset: 0;
        background: radial-gradient(ellipse 58% 40% at 50% 52%, rgba(0,0,0,.62), transparent 72%); }}
.capa {{ position: absolute; inset: 0; display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center; }}
.titulo {{ font-family: 'Cinzel', 'Trajan Pro', 'Times New Roman', serif; font-weight: 400;
          color: #fff; font-size: clamp(2.4rem, 8.6vw, 8.6rem); line-height: 1;
          letter-spacing: .22em; margin-left: .22em;
          text-shadow: 0 0 40px rgba(0,0,0,.7), 0 0 2px rgba(255,255,255,.4);
          opacity: 0; animation: qp-titulo 1.8s cubic-bezier(.16,1,.3,1) .9s forwards; }}
.linea {{ height: 1px; width: 0; margin: 1.6rem 0 1.2rem;
         background: linear-gradient(90deg, transparent, #e9c77b, transparent);
         animation: qp-linea 1.1s ease-out 1.9s forwards; }}
.sub {{ font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 300; color: rgba(255,255,255,.82);
       font-size: clamp(.7rem, 1.1vw, 1.05rem); letter-spacing: .42em; text-transform: uppercase;
       padding: 0 1rem; text-shadow: 0 0 22px rgba(0,0,0,.9);
       opacity: 0; animation: qp-aparece 1s ease-out 2.3s forwards; }}
body {{ animation: qp-fin .9s ease-in {salida:.2f}s forwards; }}
@keyframes qp-aparece {{ to {{ opacity: 1; }} }}
@keyframes qp-titulo {{ 0% {{ opacity: 0; letter-spacing: .5em; filter: blur(10px); }}
                       100% {{ opacity: 1; letter-spacing: .22em; filter: blur(0); }} }}
@keyframes qp-linea {{ to {{ width: min(420px, 55vw); }} }}
@keyframes qp-fin {{ to {{ opacity: 0; }} }}
@media (prefers-reduced-motion: reduce) {{
  #lienzo, .titulo, .sub {{ animation: none; opacity: 1; }} .linea {{ animation: none; width: 320px; }}
}}
</style></head><body>
<canvas id="lienzo"></canvas><div class="velo"></div>
<div class="capa"><div class="titulo">{titulo}</div><div class="linea"></div><div class="sub">{subtitulo}</div></div>
<script>
(function () {{
  const VERT = `{_VERT_METAL}`;
  const FRAG = `{_FRAG_METAL}`;
  const lienzo = document.getElementById("lienzo");
  const gl = lienzo.getContext("webgl", {{ alpha: true, premultipliedAlpha: true, antialias: false }});
  if (!gl) return;                                   // sin WebGL: queda el título sobre negro
  function compilar(tipo, codigo) {{
    const s = gl.createShader(tipo); gl.shaderSource(s, codigo); gl.compileShader(s);
    return gl.getShaderParameter(s, gl.COMPILE_STATUS) ? s : null;
  }}
  const vs = compilar(gl.VERTEX_SHADER, VERT), fs = compilar(gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) return;
  const prog = gl.createProgram(); gl.attachShader(prog, vs); gl.attachShader(prog, fs);
  gl.linkProgram(prog); gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
  const pos = gl.getAttribLocation(prog, "a_position");
  gl.enableVertexAttribArray(pos); gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
  const u = (n) => gl.getUniformLocation(prog, n);
  const lineal = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
  const hex = "{color_metal}".replace("#", "");
  gl.uniform3f(u("u_f0"), ...[0, 2, 4].map((i) => lineal(parseInt(hex.substr(i, 2), 16) / 255)));
  gl.uniform1f(u("u_count"), {int(cintas)});
  gl.uniform1f(u("u_iridescence"), {iridiscencia});
  gl.uniform1f(u("u_pull"), 0.0);
  gl.uniform1f(u("u_dark"), 1.0);
  gl.uniform2f(u("u_mouse"), 0.5, 0.5);
  gl.uniform2f(u("u_light"), -0.45, 0.55);

  function ajustar() {{                               // el iframe cambia de tamaño al estirarse
    const r = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.max(1, Math.floor(window.innerWidth * r));
    const h = Math.max(1, Math.floor(window.innerHeight * r));
    if (lienzo.width !== w || lienzo.height !== h) {{
      lienzo.width = w; lienzo.height = h; gl.viewport(0, 0, w, h);
    }}
  }}
  const quieto = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  function cuadro(ahora) {{
    ajustar();
    const t = 18 + (quieto ? 0 : ((ahora - t0) / 1000) * {velocidad});
    gl.uniform2f(u("u_resolution"), lienzo.width, lienzo.height);
    gl.uniform1f(u("u_time"), t);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    if (ahora - t0 < {duracion * 1000 + 500:.0f}) requestAnimationFrame(cuadro);   // se detiene al terminar
  }}
  requestAnimationFrame(cuadro);
}})();
</script></body></html>"""
