import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Root Locus Learning Laboratory", page_icon="🎓", layout="wide")

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))
FACULTY_PASSWORD = st.secrets.get("FACULTY_PASSWORD", os.getenv("FACULTY_PASSWORD", ""))
TABLE = "root_locus_learning_submissions"

def db_headers():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json", "Prefer": "return=minimal"}

def db_ok():
    return bool(SUPABASE_URL and SUPABASE_KEY)

def save_row(row):
    if not db_ok():
        st.error("Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY to Streamlit Secrets.")
        return False
    try:
        r = requests.post(f"{SUPABASE_URL.rstrip('/')}/rest/v1/{TABLE}",
                           headers=db_headers(), json=row, timeout=15)
        if r.status_code >= 300:
            st.error(f"Database error: {r.status_code} — {r.text[:400]}")
            return False
        return True
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return False

def load_rows():
    if not db_ok(): return pd.DataFrame()
    try:
        r = requests.get(f"{SUPABASE_URL.rstrip('/')}/rest/v1/{TABLE}",
                         headers=db_headers(),
                         params={"select":"*", "order":"submitted_at.desc", "limit":"3000"},
                         timeout=15)
        return pd.DataFrame(r.json()) if r.status_code < 300 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def parse_coeffs(s):
    a = np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=float)
    if len(a) == 0 or abs(a[0]) < 1e-12:
        raise ValueError("Enter valid polynomial coefficients.")
    return a

def closed_poles(num, den, K):
    n = max(len(num), len(den))
    N = np.pad(num, (n-len(num), 0))
    D = np.pad(den, (n-len(den), 0))
    return np.roots(D + K*N)

def complex_text(z):
    if abs(z.imag) < 1e-7: return f"{z.real:.4f}"
    return f"{z.real:.4f} {'+' if z.imag >= 0 else '-'} j{abs(z.imag):.4f}"

def poly_str(c):
    n = len(c)-1
    out=[]
    for i,a in enumerate(c):
        p=n-i
        if abs(a)<1e-12: continue
        sign="+" if a>=0 else "-"
        m=abs(a)
        if p==0: body=f"{m:g}"
        elif p==1: body=("" if abs(m-1)<1e-12 else f"{m:g}")+"s"
        else: body=("" if abs(m-1)<1e-12 else f"{m:g}")+f"s^{p}"
        if not out: out.append(("" if a>=0 else "-")+body)
        else: out.append(f" {sign} {body}")
    return "".join(out)

def real_axis_segments(poles, zeros):
    pts=sorted([float(x.real) for x in list(poles)+list(zeros) if abs(x.imag)<1e-7])
    if not pts: return []
    bounds=[-np.inf]+pts+[np.inf]
    seg=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        test = (b-1 if np.isinf(b) else a+1 if np.isinf(a) else (a+b)/2)
        count=sum(x.real>test+1e-9 and abs(x.imag)<1e-7 for x in list(poles)+list(zeros))
        if count%2==1: seg.append((a,b))
    return seg

def breakaway_points(num, den):
    # K(s) = -D(s)/N(s); candidates from dK/ds = 0.
    if len(num) == 1:
        return []
    D=np.poly1d(den); N=np.poly1d(num)
    P=-(D.deriv()*N-D*N.deriv())
    roots=np.roots(P)
    poles=np.roots(den); zeros=np.roots(num)
    candidates=[]
    for r in roots:
        if abs(r.imag)>1e-6: continue
        s=float(r.real)
        if abs(np.polyval(num,s))<1e-8: continue
        K=float(-np.polyval(den,s)/np.polyval(num,s))
        # candidate is meaningful for positive K and on a real-axis locus segment
        segs=real_axis_segments(poles,zeros)
        on=False
        for a,b in segs:
            if (np.isinf(a) or s>a-1e-6) and (np.isinf(b) or s<b+1e-6):
                on=True
        if K>1e-7 and on:
            candidates.append((s,K))
    return sorted(candidates)

def asymptotes(poles, zeros):
    n,m=len(poles),len(zeros)
    q=n-m
    if q<=0: return []
    centroid=(sum(p.real for p in poles)-sum(z.real for z in zeros))/q
    angles=[(2*k+1)*180/q for k in range(q)]
    return centroid, angles

def locus(num, den, kmax):
    ks=np.linspace(0,kmax,601)
    L=np.array([closed_poles(num,den,k) for k in ks])
    return ks,L

def stable(p):
    return bool(np.all(p.real < -1e-7))

def stable_range(num,den,kmax):
    ks=np.linspace(0,kmax,1501)
    ok=np.array([stable(closed_poles(num,den,k)) for k in ks])
    ranges=[]; start=None
    for i,v in enumerate(ok):
        if v and start is None: start=ks[i]
        if start is not None and (not v or i==len(ok)-1):
            end=ks[i] if v and i==len(ok)-1 else ks[i-1]
            ranges.append((float(start),float(end))); start=None
    return ranges

def plot_locus(num,den,K,kmax,show_guides=True):
    ks,L=locus(num,den,kmax)
    op,oz=np.roots(den),np.roots(num)
    cp=closed_poles(num,den,K)
    fig=go.Figure()
    for b in range(L.shape[1]):
        fig.add_trace(go.Scatter(x=L[:,b].real,y=L[:,b].imag,mode="lines",
                                 name=f"Branch {b+1}",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=op.real,y=op.imag,mode="markers",
                             marker=dict(symbol="x",size=14),name="Open-loop poles",
                             hovertext=[complex_text(z) for z in op],hoverinfo="text"))
    if len(oz):
        fig.add_trace(go.Scatter(x=oz.real,y=oz.imag,mode="markers",
                                 marker=dict(symbol="circle-open",size=14),name="Open-loop zeros",
                                 hovertext=[complex_text(z) for z in oz],hoverinfo="text"))
    fig.add_trace(go.Scatter(x=cp.real,y=cp.imag,mode="markers+text",
                             text=[f"p{i+1}" for i in range(len(cp))],textposition="top center",
                             marker=dict(symbol="diamond",size=12),name=f"Current poles K={K:g}"))
    fig.add_shape(type="line",x0=0,x1=0,y0=0,y1=1,xref="x",yref="paper",
                  line=dict(color="gray",dash="dash"))
    fig.add_shape(type="line",x0=0,x1=1,y0=0,y1=0,xref="paper",yref="y",
                  line=dict(color="gray",dash="dash"))
    if show_guides:
        for zeta in [0.2,0.4,0.6,0.8]:
            x=np.linspace(-10,0,120); th=np.arccos(zeta); y=-x*np.tan(th)
            fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line=dict(dash="dot",width=1),
                                     opacity=.3,showlegend=False))
            fig.add_trace(go.Scatter(x=x,y=-y,mode="lines",line=dict(dash="dot",width=1),
                                     opacity=.3,showlegend=False))
    xx=np.concatenate([L.real.ravel(),op.real,cp.real]); yy=np.concatenate([L.imag.ravel(),op.imag,cp.imag])
    span=max(1,float(np.max(np.abs(xx))),float(np.max(np.abs(yy))))
    fig.update_layout(height=500,margin=dict(l=15,r=15,t=45,b=15),
                      title="Interactive Root Locus — verify your predictions here",
                      xaxis_title="Real axis σ",yaxis_title="Imaginary axis jω",
                      xaxis=dict(range=[-1.2*span,1.2*span]),
                      yaxis=dict(range=[-1.2*span,1.2*span],scaleanchor="x",scaleratio=1))
    return fig

st.markdown("""<style>
.block-container{max-width:1250px;padding-top:1rem}
h1{font-size:clamp(1.7rem,5vw,2.6rem)!important}
.stButton>button,.stDownloadButton>button{width:100%;min-height:44px}
.step{padding:.6rem .8rem;border-radius:10px;border:1px solid #ddd;margin:.35rem 0}
</style>""",unsafe_allow_html=True)

st.title("🎓 Root Locus Learning Laboratory")
st.caption("Guided discovery • Interactive simulation • Online assessment • Mobile friendly")

if not db_ok():
    st.warning("Demo mode: configure Supabase before public deployment.")

page=st.radio("Navigation",["🧪 Guided Experiment","📘 Student Workbook","👨‍🏫 Faculty Dashboard","ℹ️ About"],horizontal=True,label_visibility="collapsed")

if page=="🧪 Guided Experiment":
    st.subheader("Experiment 2 — Discovering the Root Locus Method")
    st.markdown("**Learning philosophy:** Predict → Calculate → Reveal → Simulate → Verify → Reflect.")

    preset=st.selectbox("Select the system for this guided experiment",[
        "Canonical: K / [s(s+2)(s+5)]",
        "Second-order: K / [s(s+2)]",
        "System with a zero: K(s+2) / [s(s+1)(s+5)]",
        "Custom transfer function"
    ])
    presets={
        "Canonical: K / [s(s+2)(s+5)]":("1","1,7,10,0"),
        "Second-order: K / [s(s+2)]":("1","1,2,0"),
        "System with a zero: K(s+2) / [s(s+1)(s+5)]":("1,2","1,6,5,0")
    }
    dn,dd=presets.get(preset,("1","1,7,10,0"))
    c1,c2=st.columns(2)
    with c1: nt=st.text_input("Numerator N(s) coefficients",value=dn)
    with c2: dt=st.text_input("Denominator D(s) coefficients",value=dd)
    try: num,den=parse_coeffs(nt),parse_coeffs(dt)
    except Exception as e: st.error(str(e)); st.stop()
    poles,zeros=np.roots(den),np.roots(num)
    if len(den)<=len(num): st.warning("For the usual root-locus method, denominator order should exceed numerator order.")
    st.latex(r"G(s)H(s)=K\frac{"+poly_str(num)+r"}{"+poly_str(den)+r"}")

    tabs=st.tabs([
        "1️⃣ Poles & Zeros","2️⃣ Real-Axis Rule","3️⃣ Asymptotes","4️⃣ Breakaway","5️⃣ Stability","6️⃣ Simulate & Verify"
    ])

    with tabs[0]:
        st.markdown("### Step 1 — Locate the open-loop poles and zeros")
        st.write("**Your task:** inspect the transfer function and predict where poles and zeros should appear.")
        guess_poles=st.text_input("Your predicted pole locations (comma separated)")
        guess_zeros=st.text_input("Your predicted zero locations (comma separated; leave blank if none)")
        if st.button("Reveal poles and zeros",key="reveal_pz"):
            st.success("Open-loop poles: "+", ".join(complex_text(x) for x in poles))
            st.success("Open-loop zeros: "+(", ".join(complex_text(x) for x in zeros) if len(zeros) else "None"))
            st.caption("At K = 0, the closed-loop poles begin at these open-loop poles.")

    with tabs[1]:
        st.markdown("### Step 2 — Discover the real-axis rule")
        st.write("A real-axis point belongs to the root locus when the number of real open-loop poles and zeros to its right is odd.")
        segments=real_axis_segments(poles,zeros)
        user_seg=st.text_input("Predict the real-axis root-locus segments (example: -5 to -2)")
        if st.button("Reveal real-axis segments",key="reveal_real"):
            labels=[]
            for a,b in segments:
                left = "-∞" if np.isneginf(a) else f"{a:g}"
                right = "∞" if np.isposinf(b) else f"{b:g}"
                labels.append(f"{left} to {right}")
            st.success("Real-axis segments: "+(", ".join(labels) if labels else "None"))

    with tabs[2]:
        st.markdown("### Step 3 — Find asymptotes and centroid")
        q=len(poles)-len(zeros)
        st.write(f"Number of branches going to infinity = **n − m = {q}**.")
        pred_cent=st.number_input("Predict centroid σₐ (optional)",value=0.0,step=0.5)
        if st.button("Reveal asymptotes",key="reveal_asym"):
            result=asymptotes(poles,zeros)
            if result:
                centroid,angles=result
                st.success(f"Centroid = {centroid:.4f}")
                st.success("Asymptote angles = "+", ".join(f"{a:g}°" for a in angles))
            else:
                st.info("No asymptotes are required because n − m ≤ 0.")

    with tabs[3]:
        st.markdown("### Step 4 — Investigate breakaway / break-in points")
        st.write("For real-axis candidates, use K(s) = −D(s)/N(s) and solve dK/ds = 0. Only candidates giving positive K and lying on a real-axis locus segment are meaningful.")
        ba=breakaway_points(num,den)
        if st.button("Reveal breakaway / break-in points",key="reveal_ba"):
            if ba:
                df=pd.DataFrame({"s candidate":[round(x,5) for x,k in ba],"K":[round(k,5) for x,k in ba]})
                st.success("Candidate breakaway/break-in points:")
                st.dataframe(df,use_container_width=True,hide_index=True)
            else:
                st.info("No positive-K breakaway/break-in candidate was found for this system.")

    with tabs[4]:
        st.markdown("### Step 5 — Predict the stable gain range")
        st.write("Continuous-time closed-loop stability requires every closed-loop pole to remain in the left-half s-plane.")
        user_range=st.text_input("Your predicted stable K range")
        if st.button("Reveal numerical stability range",key="reveal_stab"):
            ranges=stable_range(num,den,200)
            if ranges:
                st.success("Numerically scanned stable region(s): "+", ".join(f"{a:g} ≤ K ≤ {b:g}" for a,b in ranges))
            else: st.warning("No stable region found for 0 ≤ K ≤ 200.")

    with tabs[5]:
        st.markdown("### Step 6 — Simulate and verify your predictions")
        K=st.slider("Move K and watch the poles travel",0.0,200.0,5.0,0.1)
        cp=np.array(closed_poles(num,den,K))
        st.plotly_chart(plot_locus(num,den,K,200,True),use_container_width=True)
        st.write("**Current closed-loop poles:**", ", ".join(complex_text(z) for z in cp))
        st.metric("Current stability", "STABLE" if stable(cp) else "UNSTABLE / BOUNDARY")
        st.info("Challenge: move K until a pair of poles approaches the imaginary axis. Record the approximate K and compare it with your Step 5 prediction.")

        st.markdown("---")
        st.subheader("🧠 Concept Check")
        with st.form("concept_check"):
            a,b=st.columns(2)
            with a:
                q1=st.radio("1. At K = 0, closed-loop poles are located at:",["Open-loop poles","Open-loop zeros","Origin only"])
                q2=st.radio("2. For real-coefficient systems, the root locus is symmetric about:",["Real axis","Imaginary axis","Neither"])
            with b:
                q3=st.radio("3. Stable continuous-time poles must lie in:",["LHP","RHP","Only on imaginary axis"])
                q4=st.radio("4. Increasing K causes closed-loop poles to:",["Move along root-locus branches","Remain fixed","Become zeros"])
            check=st.form_submit_button("Check my answers")
            if check:
                answers=[q1=="Open-loop poles",q2=="Real axis",q3=="LHP",q4=="Move along root-locus branches"]
                score=sum(answers)
                if score==4: st.success("4/4 correct — excellent.")
                else: st.warning(f"{score}/4 correct. Revisit the relevant step and try again.")

        st.subheader("📝 Submit Guided Experiment")
        with st.form("submission",clear_on_submit=True):
            a,b=st.columns(2)
            with a: name=st.text_input("Student name *")
            with b: roll=st.text_input("Roll number / ID *")
            prediction=st.text_area("1. What did you predict about pole movement?")
            real_rule=st.text_area("2. Explain the real-axis rule in your own words.")
            asym=st.text_area("3. What did you learn from the asymptotes/centroid?")
            breakaway=st.text_area("4. What did you learn from breakaway analysis?")
            stability=st.text_area("5. State your predicted and verified stable K range.")
            reflection=st.text_area("6. What engineering insight did this experiment give you?")
            submit=st.form_submit_button("Submit learning experiment",type="primary")
            if submit:
                if not name.strip() or not roll.strip(): st.error("Enter student name and roll number.")
                else:
                    ranges=stable_range(num,den,200)
                    row={
                        "submitted_at":datetime.now(timezone.utc).isoformat(),
                        "student_name":name.strip(),"roll_no":roll.strip(),
                        "experiment":"Guided Root Locus Learning Experiment",
                        "numerator":",".join(f"{x:g}" for x in num),
                        "denominator":",".join(f"{x:g}" for x in den),
                        "predicted_poles":guess_poles,"predicted_zeros":guess_zeros,
                        "predicted_real_axis":user_seg,"predicted_centroid":float(pred_cent),
                        "predicted_stable_range":user_range,
                        "prediction":prediction.strip(),"real_axis_explanation":real_rule.strip(),
                        "asymptote_learning":asym.strip(),"breakaway_learning":breakaway.strip(),
                        "stability_conclusion":stability.strip(),"reflection":reflection.strip(),
                        "verified_stable_region":", ".join(f"{a:g} to {b:g}" for a,b in ranges) if ranges else "None",
                        "concept_score":None
                    }
                    if save_row(row): st.success("Submitted successfully. The guided workbook is cleared for the next student.")



elif page=="📘 Student Workbook":
    st.subheader("📘 Guided Root Locus Workbook")
    df=load_rows()
    st.dataframe(df,use_container_width=True,hide_index=True) if not df.empty else st.info("No submissions yet.")

elif page=="👨‍🏫 Faculty Dashboard":
    st.subheader("👨‍🏫 Faculty Dashboard — Guided Root Locus")
    if FACULTY_PASSWORD:
        pw=st.text_input("Faculty password",type="password")
        if pw!=FACULTY_PASSWORD: st.info("Enter the faculty password."); st.stop()
    else: st.warning("Set FACULTY_PASSWORD in Streamlit Secrets.")
    df=load_rows()
    if df.empty: st.info("No submissions found.")
    else:
        a,b,c=st.columns(3)
        a.metric("Submissions",len(df)); b.metric("Students",df.roll_no.nunique()); c.metric("Experiments",df.experiment.nunique())
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download guided-lab CSV",df.to_csv(index=False).encode(),
                           "root_locus_learning_submissions.csv","text/csv")

else:
    st.subheader("About this learning laboratory")
    st.markdown("""
### From plotting to learning

The student is deliberately **not given every answer immediately**. Each stage follows:

**Predict → Calculate → Reveal → Simulate → Verify → Reflect**

### Guided stages

1. **Poles & zeros** — identify starting and ending locations.
2. **Real-axis rule** — predict which real-axis segments belong to the locus.
3. **Asymptotes & centroid** — calculate branch directions.
4. **Breakaway / break-in** — investigate where branches leave or meet the real axis.
5. **Stability** — predict the stable K region.
6. **Simulation** — move K and verify the prediction.
7. **Concept check** — immediate formative feedback.
8. **Digital workbook** — submit reasoning, not just numerical answers.

This is intended to function as an actual **Control Systems teaching laboratory**.
""")
