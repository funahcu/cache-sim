import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# ─── ページ設定 ───────────────────────────────────────────────
st.set_page_config(page_title="Cache simulator", page_icon="🕸️", layout="centered")

# ─── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background:#0a0a0f;color:#e8e8f0;}
h1,h2,h3{font-family:'Syne',sans-serif!important;font-weight:800!important;letter-spacing:-0.02em;}
.block-container{padding-top:2rem;max-width:780px;}
.stSlider label{font-family:'Space Mono',monospace!important;font-size:0.76rem!important;
    color:#8888aa!important;text-transform:uppercase;letter-spacing:0.08em;}
.model-badge{display:inline-block;padding:3px 12px;border-radius:20px;
    font-family:'Space Mono',monospace;font-size:0.70rem;font-weight:700;
    letter-spacing:0.1em;text-transform:uppercase;}
.badge-ba{background:#7c6af7;color:#fff;}
.badge-sf{background:#06b6d4;color:#fff;}
.info-card{background:#12121e;border:1px solid #2a2a3e;border-radius:10px;
    padding:.9rem 1.2rem;font-family:'Space Mono',monospace;font-size:0.76rem;
    color:#8888aa;margin-bottom:.8rem;}
.info-card span{color:#c4b5fd;font-weight:700;}
.hint{font-family:'Space Mono',monospace;font-size:0.70rem;color:#444466;
    margin-top:-0.3rem;margin-bottom:.7rem;}
.path-card{background:#0d0d1a;border:1px solid #1e1e3a;border-radius:10px;
    padding:1rem 1.4rem;font-family:'Space Mono',monospace;font-size:0.80rem;
    color:#aaaacc;margin-top:.8rem;line-height:2.0;}
.path-title{color:#8888ff;font-weight:700;font-size:0.83rem;
    margin-bottom:.5rem;letter-spacing:.04em;}
.no-path{color:#cc4444;}
.legend-row{display:flex;gap:1.4rem;flex-wrap:wrap;margin-bottom:.5rem;
    font-family:'Space Mono',monospace;font-size:0.72rem;color:#8888aa;}
.legend-dot{display:inline-block;width:11px;height:11px;border-radius:50%;
    margin-right:4px;vertical-align:middle;}
</style>
""", unsafe_allow_html=True)

# ─── 定数 ────────────────────────────────────────────────────
STATES   = ["Nothing", "Orig", "Cache"]   # ローテート順
S_COLORS = {                              # 塗り色
    "Nothing": "rgba(0,0,0,0)",
    "Orig":    "#ff6644",
    "Cache":   "#44ddaa",
}
S_BORDER = {
    "Nothing": "#6655aa",
    "Orig":    "#ffaa88",
    "Cache":   "#88ffcc",
}
S_TEXT = {
    "Nothing": "#9988cc",
    "Orig":    "#ffffff",
    "Cache":   "#ffffff",
}

# ─── セッション初期化 ─────────────────────────────────────────
def _init():
    defs = dict(model_type="BA Model", node_states={}, graph_edges=[],
                graph_pos={}, graph_drawn=False, last_n_nodes=0,
                last_n_links=0, last_model="", source_node=0,
                target_type="Orig", regen_seed=42)
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ─── タイトル ────────────────────────────────────────────────
st.markdown("## 🕸️ Cache Simulator")
st.markdown("<p style='font-family:Space Mono,monospace;font-size:0.78rem;"
            "color:#555577;margin-top:-.5rem;'>"
            "BA / Scale-Free · Orig / Cache / Nothing · shortest path finder</p>",
            unsafe_allow_html=True)
st.divider()

# ─── スライダー ──────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    n_nodes = st.slider("Node count", 1, 20, 10, 1)
with c2:
    max_links = n_nodes*(n_nodes-1)//2 if n_nodes>1 else 1
    min_links = n_nodes if n_nodes <= max_links else max_links
    pv = max(min_links, min(st.session_state.get("_lv", min_links), max_links))
    n_links = st.slider("Link count", min_links, max(min_links,max_links), pv, 1, key="_lv")

# ─── モデルボタン ────────────────────────────────────────────
st.markdown("")
cb1, cb2, _ = st.columns([1,1,2])
with cb1:
    if st.button("⬡  BA Model", use_container_width=True,
                 type="primary" if st.session_state.model_type=="BA Model" else "secondary"):
        st.session_state.model_type = "BA Model"
with cb2:
    if st.button("✦  Scale-Free", use_container_width=True,
                 type="primary" if st.session_state.model_type=="Scale-Free" else "secondary"):
        st.session_state.model_type = "Scale-Free"

bc = "badge-ba" if st.session_state.model_type=="BA Model" else "badge-sf"
st.markdown(f"<p style='margin-top:.4rem;'>現在のモデル: "
            f"<span class='model-badge {bc}'>{st.session_state.model_type}</span></p>",
            unsafe_allow_html=True)

# ─── グラフ生成 ──────────────────────────────────────────────
def _adjust_edges(G, target, seed):
    rng   = np.random.default_rng(seed)
    nodes = list(G.nodes())
    for _ in range(50000):
        if G.number_of_edges() >= target: break
        u, v = rng.choice(nodes, 2, replace=False)
        if not G.has_edge(u,v): G.add_edge(u,v)
    edges = list(G.edges()); rng.shuffle(edges)
    for e in edges:
        if G.number_of_edges() <= target: break
        G.remove_edge(*e)
    return G

def build_ba(n, target, seed):
    if n<=1:
        G=nx.Graph(); G.add_nodes_from(range(n)); return G
    best_G, best_d = None, float("inf")
    for m in range(1,n):
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        d = abs(G.number_of_edges()-target)
        if d < best_d: best_d, best_G = d, G
        if G.number_of_edges() >= target: break
    return _adjust_edges(best_G.copy(), target, seed)

def build_sf(n, target, seed):
    if n<=2:
        G=nx.Graph(); G.add_nodes_from(range(n))
        if n==2: G.add_edge(0,1)
        return G
    best_G, best_d = None, float("inf")
    for m in range(1,n):
        for p in [0.0,0.3,0.6,0.9]:
            try:
                G = nx.powerlaw_cluster_graph(n,m,p,seed=seed)
                d = abs(G.number_of_edges()-target)
                if d<best_d: best_d,best_G = d,G
            except: pass
    G = best_G.copy() if best_G else nx.path_graph(n)
    return _adjust_edges(G, target, seed)

def layout(G):
    n = G.number_of_nodes()
    if n<=3 or G.number_of_edges()==0: return nx.circular_layout(G)
    try:    return nx.kamada_kawai_layout(G)
    except: return nx.spring_layout(G,seed=42,k=2.5/n**0.5,iterations=120)

# ─── 最短経路 ────────────────────────────────────────────────
def shortest_paths(edges, n_total, node_states, source, target_type):
    G = nx.Graph(); G.add_nodes_from(range(n_total)); G.add_edges_from(edges)

    def is_target(s):
        return s in ("Orig","Cache") if target_type=="Cache or Orig" else s==target_type

    if source == "all":
        # 各ソースノードから最も近いターゲット1件のみ：(src, tgt, path) のリスト
        targets = sorted([i for i,s in node_states.items() if is_target(s)])
        results = []
        for src in range(n_total):
            if src in targets:
                # 自身がターゲット → 0ホップで確定
                results.append((src, src, [src]))
                continue
            best_path, best_tgt, best_len = None, None, float("inf")
            for tgt in targets:
                try:
                    p = nx.shortest_path(G, src, tgt)
                    if len(p) < best_len:
                        best_len, best_tgt, best_path = len(p), tgt, p
                except:
                    pass
            results.append((src, best_tgt, best_path))  # best_tgt/path は None の場合もあり
        return results
    else:
        # 単一ソース
        # ソース自身がターゲット状態なら 0ホップとして先頭に追加
        results = []
        if is_target(node_states.get(source, "Nothing")):
            results.append((source, [source]))
        targets = [i for i,s in node_states.items() if is_target(s) and i!=source]
        for t in sorted(targets):
            try:    results.append((t, nx.shortest_path(G, source, t)))
            except: results.append((t, None))
        return results

# ─── Plotly図 ────────────────────────────────────────────────
def make_fig(edges, pos, node_states, n_total, model_name,
             source, highlight_paths):
    is_ba  = model_name=="BA Model"
    accent = "#a78bfa" if is_ba else "#22d3ee"
    bg     = "#0a0a0f"

    path_edges = set()
    path_nodes = set()
    for path in highlight_paths:
        path_nodes.update(path)
        for a,b in zip(path,path[1:]):
            path_edges.add((min(a,b),max(a,b)))

    deg = {i:0 for i in range(n_total)}
    for u,v in edges: deg[u]+=1; deg[v]+=1
    max_d = max(deg.values()) if deg else 1

    # エッジ
    ex_n,ey_n,ex_h,ey_h = [],[],[],[]
    for u,v in edges:
        x0,y0=pos[u]; x1,y1=pos[v]
        key=(min(u,v),max(u,v))
        if key in path_edges: ex_h+=[x0,x1,None]; ey_h+=[y0,y1,None]
        else:                 ex_n+=[x0,x1,None]; ey_n+=[y0,y1,None]

    traces = [go.Scatter(x=ex_n,y=ey_n,mode="lines",
                         line=dict(width=2.0,color=accent),opacity=0.5,hoverinfo="none")]
    if ex_h:
        traces.append(go.Scatter(x=ex_h,y=ey_h,mode="lines",
                                 line=dict(width=3.5,color="#ffdd44"),opacity=0.80,hoverinfo="none"))

    # ノードをバケツ分け (source / Orig / Cache / Nothing+waypoint / Nothing)
    buckets = {"source":[], "Orig":[], "Cache":[], "waypoint":[], "Nothing":[]}
    for i in range(n_total):
        x,y=pos[i]; sz=22+30*(deg[i]/max(max_d,1))**0.6
        st_ = node_states.get(i,"Nothing")
        if i==source:       buckets["source"].append((i,x,y,sz,st_))
        elif st_=="Orig":   buckets["Orig"].append((i,x,y,sz,st_))
        elif st_=="Cache":  buckets["Cache"].append((i,x,y,sz,st_))
        elif i in path_nodes: buckets["waypoint"].append((i,x,y,sz,st_))
        else:               buckets["Nothing"].append((i,x,y,sz,st_))

    def mk_trace(items, fill, border, tcolor, name, symbol="circle", extra_border=None):
        if not items: return None
        return go.Scatter(
            x=[it[1] for it in items], y=[it[2] for it in items],
            mode="markers+text",
            marker=dict(size=[it[3] for it in items], color=fill, symbol=symbol,
                        line=dict(width=2.5, color=border if not extra_border else extra_border),
                        opacity=0.95),
            text=[str(it[0]) for it in items],
            textposition="middle center",
            textfont=dict(size=20,color=tcolor,family="Space Mono"),
            hovertemplate=f"Node %{{text}}  [{name}]<extra></extra>",
            name=name,
        )

    # source: 星 (single) or 経路出発ノード群 (all)
    if source == "all":
        path_sources = sorted(set(path[0] for path in highlight_paths if path and len(path)>1))
        if path_sources:
            ps_items = [(i, *pos[i], 22+30*(deg[i]/max(max_d,1))**0.6,
                         node_states.get(i,"Nothing")) for i in path_sources]
            traces.append(go.Scatter(
                x=[it[1] for it in ps_items], y=[it[2] for it in ps_items],
                mode="markers+text",
                marker=dict(size=[it[3]+6 for it in ps_items], color="rgba(0,0,0,0)",
                            symbol="star", line=dict(width=2.5, color="#ff8844"), opacity=0.90),
                text=[str(it[0]) for it in ps_items],
                textposition="middle center",
                textfont=dict(size=10, color="#ff8844", family="Space Mono"),
                hovertemplate="Node %{text} [path source]<extra></extra>",
                name="PATH_SRC"))
    else:
        si = buckets["source"]
        if si:
            i,x,y,sz,st_ = si[0]
            fill  = S_COLORS[st_]
            bord  = "#ff8844"
            tclr  = S_TEXT[st_] if st_!="Nothing" else "#ff8844"
            traces.append(go.Scatter(x=[x],y=[y],mode="markers+text",
                marker=dict(size=sz+10,color=fill,symbol="star",
                            line=dict(width=3,color=bord),opacity=0.98),
                text=[str(i)],textposition="middle center",
                textfont=dict(size=11,color=tclr,family="Space Mono"),
                hovertemplate=f"Node {i} [SOURCE / {st_}]<extra></extra>",name="SOURCE"))

    t=mk_trace(buckets["Orig"],  S_COLORS["Orig"],  S_BORDER["Orig"],  S_TEXT["Orig"],  "Orig")
    if t: traces.append(t)
    t=mk_trace(buckets["Cache"], S_COLORS["Cache"], S_BORDER["Cache"], S_TEXT["Cache"], "Cache")
    if t: traces.append(t)
    # waypoint (Nothing だが経路上)
    t=mk_trace(buckets["waypoint"],"rgba(255,221,68,0.15)","#ffdd44","#ffdd44","waypoint")
    if t: traces.append(t)
    # 通常 Nothing
    t=mk_trace(buckets["Nothing"],"rgba(0,0,0,0)",accent,S_TEXT["Nothing"],"Nothing")
    if t: traces.append(t)

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=bg,plot_bgcolor=bg,
        margin=dict(l=10,r=10,t=50,b=10),height=540,
        title=dict(
            text=(f"{model_name}  ·  {n_total} nodes / {len(edges)} edges  "
                  f"<span style='font-size:11px;color:#555577;'>"
                  f"— click = rotate state · ★ = source</span>"),
            font=dict(size=13,color=accent,family="Space Mono"),x=0.02),
        showlegend=False,
        xaxis=dict(visible=False),yaxis=dict(visible=False),dragmode="pan",
    )
    return fig

# ─── 経路テキスト ─────────────────────────────────────────────
def render_paths(results, source, target_type, node_states):
    tcolor = {"Orig":"#ff6644","Cache":"#44ddaa","Cache or Orig":"#ffcc44","Nothing":"#9988cc"}
    tc     = tcolor.get(target_type, "#ffcc44")
    is_all = source == "all"

    if not is_all:
        src_st    = node_states.get(source, "Nothing")
        src_label = f"★Node {source} [{src_st}]"
    else:
        src_label = "All nodes (static)"

    if not results or (is_all and all(tgt is None for _, tgt, *_ in results)):
        def _any_target(i, s):
            if target_type == "Cache or Orig": return s in ("Orig","Cache")
            return s == target_type
        targets_exist = any(_any_target(i,s) for i,s in node_states.items()
                            if (True if is_all else i!=source))
        msg = (f"<span class='no-path'>{target_type} ノードがありません。</span>"
               if not targets_exist else
               f"<span class='no-path'>到達可能な {target_type} ノードがありません。</span>")
        st.markdown(f"<div class='path-card'>"
                    f"<div class='path-title'>📡 Paths to <span style='color:{tc}'>{target_type}</span> "
                    f"from {src_label}</div>{msg}</div>", unsafe_allow_html=True)
        return

    def fmt_path(src_node, tgt, path):
        """(src, tgt, path) を1行のHTML文字列に変換"""
        if tgt is None:
            # ターゲット自体が存在しない（全ノードがNothing等）
            return (f"<span style='color:#ff8844;font-weight:700'>★{src_node}</span>"
                    f" <span style='color:#cc4444'>(ターゲットなし)</span>")
        if path is None:
            return (f"<span style='color:#ff8844;font-weight:700'>★{src_node}</span>"
                    f"<span style='color:#44dd44'> → </span>"
                    f"<span style='color:{tc}'>◆{tgt}</span>"
                    f" <span style='color:#cc4444'>(経路なし)</span>")
        hops = len(path) - 1
        if hops == 0:
            return (f"<span style='color:{tc};font-weight:700'>◆{tgt}</span>"
                    f" <span style='color:#444466;font-size:.73rem;'>(0 hops — self)</span>")
        parts = []
        for nid in path:
            if nid == src_node:
                c = "#ff8844"; lbl = f"★{nid}"
            elif nid == tgt:
                c = tc; lbl = f"◆{nid}"
            else:
                c = "#ccccee"; lbl = str(nid)
            parts.append(f"<span style='color:{c};font-weight:700'>{lbl}</span>")
        arr = "<span style='color:#44dd44'> → </span>"
        return arr.join(parts) + (f" <span style='color:#444466;font-size:.73rem;'>"
                                   f"({hops} hop{'s' if hops!=1 else ''})</span>")

    if not is_all:
        # 単一ソース: フラットなリスト (tgt, path)
        lines = [fmt_path(source, tgt, path) for tgt, path in results]
        st.markdown(
            f"<div class='path-card'>"
            f"<div class='path-title'>📡 Shortest Paths &nbsp; {src_label} &nbsp;→&nbsp; "
            f"<span style='color:{tc}'>{target_type}</span> nodes</div>"
            + "<br>".join(lines) + "</div>",
            unsafe_allow_html=True)
    else:
        # All (static): ソースノードごとに1行、最近傍ターゲット1件のみ表示
        lines = []
        for src_node, tgt, path in results:
            src_st = node_states.get(src_node, "Nothing")
            sc     = {"Orig":"#ff6644","Cache":"#44ddaa"}.get(src_st, "#ff8844")
            src_span = (f"<span style='color:#ff8844;font-weight:700'>★{src_node}</span>"
                        f"<span style='color:{sc};font-size:.74rem;'>[{src_st}]</span>")
            lines.append(src_span + "  " + fmt_path(src_node, tgt, path))

        st.markdown(
            f"<div class='path-card'>"
            f"<div class='path-title'>📡 All nodes (static) &nbsp;→&nbsp; "
            f"<span style='color:{tc}'>{target_type}</span> (nearest)</div>"
            + "<br>".join(lines) + "</div>",
            unsafe_allow_html=True)

# ─── Draw / Regen ─────────────────────────────────────────────
st.divider()
st.markdown(f"""<div class="info-card">
    Nodes: <span>{n_nodes}</span> &nbsp;|&nbsp;
    Links: <span>{n_links}</span> &nbsp;|&nbsp;
    Model: <span>{st.session_state.model_type}</span>
    </div>""", unsafe_allow_html=True)

db1, db2 = st.columns([2,1])
with db1:
    draw_clicked = st.button("▶  Draw Network", type="primary", use_container_width=True)
with db2:
    regen_clicked = st.button("🔀  Re-generate", use_container_width=True,
                              disabled=not st.session_state.graph_drawn)

st.markdown('<p class="hint">💡 ノードクリック: Nothing → Orig → Cache → Nothing …　'
            '｜　Re-generate: 同条件で別グラフを生成</p>', unsafe_allow_html=True)

def do_generate(seed):
    model = st.session_state.model_type
    G = build_ba(n_nodes,n_links,seed) if model=="BA Model" else build_sf(n_nodes,n_links,seed)
    pos = layout(G)
    st.session_state.graph_edges  = list(G.edges())
    st.session_state.graph_pos    = {i:list(pos[i]) for i in G.nodes()}
    st.session_state.node_states  = {i:"Nothing" for i in G.nodes()}
    st.session_state.graph_drawn  = True
    st.session_state.last_n_nodes = n_nodes
    st.session_state.last_n_links = n_links
    st.session_state.last_model   = model
    st.session_state.source_node  = 0
    st.session_state.regen_seed   = seed

if draw_clicked:
    with st.spinner("Generating…"):
        do_generate(42)

if regen_clicked:
    with st.spinner("Re-generating…"):
        new_seed = st.session_state.regen_seed + 1
        do_generate(new_seed)

# ─── メイン表示 ──────────────────────────────────────────────
if st.session_state.graph_drawn:
    # 旧バージョンのbool状態を文字列に正規化
    for k, v in st.session_state.node_states.items():
        if v is True:  st.session_state.node_states[k] = "Orig"
        elif v is False: st.session_state.node_states[k] = "Nothing"

    n_total  = st.session_state.last_n_nodes
    pos_dict = {i:tuple(v) for i,v in st.session_state.graph_pos.items()}

    # 凡例
    st.markdown(
        "<div class='legend-row'>"
        "<span><span class='legend-dot' style='background:#ff8844;'></span>★ Source</span>"
        "<span><span class='legend-dot' style='background:#ff6644;'></span>Orig</span>"
        "<span><span class='legend-dot' style='background:#44ddaa;'></span>Cache</span>"
        "<span><span class='legend-dot' style='border:2px solid #6655aa;background:transparent;'></span>Nothing</span>"
        "<span><span class='legend-dot' style='border:2px solid #ffdd44;background:rgba(255,221,68,.15);'></span>Waypoint</span>"
        "</div>", unsafe_allow_html=True)

    # コントロール行
    cc1, cc2, cc3 = st.columns([1.2, 1, 1])
    with cc1:
        src_options = ["all"] + list(range(n_total))
        cur_src = st.session_state.source_node
        src_idx = src_options.index(cur_src) if cur_src in src_options else 1
        src = st.selectbox(
            "出発ノード (★)", src_options, index=src_idx,
            format_func=lambda x: "All (static)" if x=="all" else f"Node {x}",
            key="_src")
        st.session_state.source_node = src
    with cc2:
        TARGET_OPTIONS = ["Orig", "Cache or Orig", "Cache"]
        if st.session_state.target_type not in TARGET_OPTIONS:
            st.session_state.target_type = "Orig"
        ttype = st.selectbox("経路ターゲット", TARGET_OPTIONS,
                             index=TARGET_OPTIONS.index(st.session_state.target_type),
                             key="_tt")
        st.session_state.target_type = ttype
    with cc3:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("All Orig", use_container_width=True):
                for k in st.session_state.node_states: st.session_state.node_states[k]="Orig"
                st.rerun()
        with col_b:
            if st.button("All None", use_container_width=True):
                for k in st.session_state.node_states: st.session_state.node_states[k]="Nothing"
                st.rerun()

    # 経路計算
    path_results   = shortest_paths(st.session_state.graph_edges, n_total,
                                    st.session_state.node_states, src, ttype)
    if src == 'all':
        highlight_paths = [p for _,_,p in path_results if p]
    else:
        highlight_paths = [p for _,p in path_results if p]

    # 図
    fig = make_fig(st.session_state.graph_edges, pos_dict,
                   st.session_state.node_states, n_total,
                   st.session_state.last_model, src, highlight_paths)

    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="net_plot")

    # クリック処理 — make_fig と同じtrace順を再現してnode_id逆引き
    if event and event.get("selection"):
        points = event["selection"].get("points",[])
        for pt in points:
            cn   = pt.get("curve_number",-1)
            pidx = pt.get("point_index")
            if pidx is None or cn<0: continue

            states = st.session_state.node_states
            path_nodes_set = set()
            for path in highlight_paths: path_nodes_set.update(path)

            # ── make_fig と同じ順でtraceを積む ──────────────────────
            # trace 0: 通常エッジ (常に1本)
            # trace 1: 強調エッジ (highlight_pathsがある場合のみ)
            has_path_edges = len(highlight_paths) > 0
            ci = 2 if has_path_edges else 1

            trace_node_lists = []

            if src == "all":
                # make_fig: source=="all" → PATH_SRC (path_sources が空でなければ)
                path_sources = sorted(set(
                    path[0] for path in highlight_paths if path and len(path) > 1))
                if path_sources:
                    trace_node_lists.append((ci, path_sources)); ci += 1
            else:
                # make_fig: source != "all" → source ノード1つ (star)
                trace_node_lists.append((ci, [src])); ci += 1

            # Orig (source以外)
            orig_l = sorted([i for i in range(n_total)
                             if states.get(i,"Nothing")=="Orig"
                             and (True if src=="all" else i!=src)])
            if orig_l: trace_node_lists.append((ci, orig_l)); ci += 1
            # Cache (source以外)
            cach_l = sorted([i for i in range(n_total)
                             if states.get(i,"Nothing")=="Cache"
                             and (True if src=="all" else i!=src)])
            if cach_l: trace_node_lists.append((ci, cach_l)); ci += 1
            # waypoint: Nothing & in path_nodes_set (source以外)
            wp_l = sorted([i for i in range(n_total)
                           if states.get(i,"Nothing")=="Nothing"
                           and i in path_nodes_set
                           and (True if src=="all" else i!=src)])
            if wp_l: trace_node_lists.append((ci, wp_l)); ci += 1
            # Nothing: それ以外
            no_l = sorted([i for i in range(n_total)
                           if states.get(i,"Nothing")=="Nothing"
                           and i not in path_nodes_set
                           and (True if src=="all" else i!=src)])
            if no_l: trace_node_lists.append((ci, no_l)); ci += 1

            for curve_ci, node_list in trace_node_lists:
                if cn == curve_ci and pidx < len(node_list):
                    nid = node_list[pidx]
                    cur = states.get(nid, "Nothing")
                    nxt = STATES[(STATES.index(cur)+1) % len(STATES)]
                    st.session_state.node_states[nid] = nxt
                    st.rerun()

    # 経路テキスト
    render_paths(path_results, src, ttype, st.session_state.node_states)

    # 統計
    cnt = {s:sum(1 for v in st.session_state.node_states.values() if v==s) for s in STATES}
    all_deg={i:0 for i in range(n_total)}
    for u,v in st.session_state.graph_edges: all_deg[u]+=1; all_deg[v]+=1
    degs=list(all_deg.values())
    st.markdown(
        f"""<div class="info-card" style="margin-top:.6rem;">
        Orig: <span>{cnt['Orig']}</span> &nbsp;|&nbsp;
        Cache: <span>{cnt['Cache']}</span> &nbsp;|&nbsp;
        Nothing: <span>{cnt['Nothing']}</span> &nbsp;‖&nbsp;
        Avg deg: <span>{np.mean(degs):.2f}</span> &nbsp;|&nbsp;
        Density: <span>{len(st.session_state.graph_edges)/max(n_total*(n_total-1)//2,1):.3f}</span>
        &nbsp;|&nbsp; seed: <span>{st.session_state.regen_seed}</span>
        </div>""", unsafe_allow_html=True)
