import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

# =============================================================================
# AYARLAR: Klasör yolunu buraya yapıştır
# =============================================================================
KLASOR_YOLU = "."

st.set_page_config(page_title="Pro İddaa Simülasyon", layout="wide")

# --- SESSION STATE BAŞLATMA ---
if 'last_ev_takim' not in st.session_state: st.session_state.last_ev_takim = None
if 'last_dep_takim' not in st.session_state: st.session_state.last_dep_takim = None

keys = ['ev_gol_inp', 'ev_yen_inp', 'ev_form_inp', 'ev_sot_inp', 
        'dep_gol_inp', 'dep_yen_inp', 'dep_form_inp', 'dep_sot_inp']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = 0.0

# --- FONKSİYONLAR ---
@st.cache_data
def dosya_yukle(dosya_yolu):
    try:
        df = pd.read_csv(dosya_yolu)
        if len(df.columns) < 2: 
            df = pd.read_csv(dosya_yolu, sep=';')
        
        df.columns = df.columns.str.strip()
        
        if 'DateTime' in df.columns: df.rename(columns={'DateTime': 'Date'}, inplace=True)
        if 'Date' not in df.columns and 'date' in df.columns: df.rename(columns={'date': 'Date'}, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')
        
        # Sayısal Dönüşümler
        numeric_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HC', 'AC', 'HST', 'AST']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- HESAPLAMALAR ---
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            df['TotalGoals'] = df['FTHG'] + df['FTAG']
            df['KG_Var'] = (df['FTHG'] > 0) & (df['FTAG'] > 0)
            df['Over15'] = df['TotalGoals'] > 1.5
            df['Over25'] = df['TotalGoals'] > 2.5
            df['Home_Over05'] = df['FTHG'] > 0.5
            df['Home_Over15'] = df['FTHG'] > 1.5
            df['Away_Over05'] = df['FTAG'] > 0.5
            df['Away_Over15'] = df['FTAG'] > 1.5
        
        if 'HTHG' in df.columns and 'HTAG' in df.columns:
            df['HT_TotalGoals'] = df['HTHG'] + df['HTAG']
            df['HT_Over05'] = df['HT_TotalGoals'] > 0.5
            df['HT_Over15'] = df['HT_TotalGoals'] > 1.5
            if 'HTR' not in df.columns:
                conditions = [(df['HTHG'] > df['HTAG']), (df['HTHG'] < df['HTAG'])]
                choices = ['H', 'A']
                df['HTR'] = np.select(conditions, choices, default='D')
        
        # KORNER HESAPLAMASI
        if 'HC' in df.columns and 'AC' in df.columns:
            df['TotalCorners'] = df['HC'] + df['AC']
            df['Corn_Over85'] = df['TotalCorners'] > 8.5
            df['Corn_Over95'] = df['TotalCorners'] > 9.5
            df['Corn_Over105'] = df['TotalCorners'] > 10.5

        # ORTALAMALAR (Simülasyon İçin - Geçmiş Veri)
        if 'FTR' in df.columns:
            df['Home_Points'] = np.where(df['FTR'] == 'H', 3, np.where(df['FTR'] == 'D', 1, 0))
            df['Away_Points'] = np.where(df['FTR'] == 'A', 3, np.where(df['FTR'] == 'D', 1, 0))
            
            # Not: Veritabanı taraması için hala Ev/Dep ayrımı kullanıyoruz ki,
            # "Evinde X ortalaması olan takımlar" gibi spesifik filtreleme yapabilelim.
            # Ancak kullanıcı girdisi artık GENEL ortalama olacak.
            df['Home_Scored_Avg'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            df['Away_Scored_Avg'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            
            df['Home_Conceded_Avg'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            df['Away_Conceded_Avg'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            
            # Şut Ortalaması
            if 'HST' in df.columns and 'AST' in df.columns:
                df['Home_SOT_Avg'] = df.groupby('HomeTeam')['HST'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
                df['Away_SOT_Avg'] = df.groupby('AwayTeam')['AST'].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean())
            
            df['Home_Form_5'] = df.groupby('HomeTeam')['Home_Points'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=3).sum())
            df['Away_Form_5'] = df.groupby('AwayTeam')['Away_Points'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=3).sum())

        return df
    except Exception as e:
        return None

def get_genel_istatistik(df, takim_adi, mac_sayisi=10):
    """
    Bir takımın hem iç hem dış sahadaki son N maçının ortalamalarını hesaplar.
    """
    # Takımın olduğu tüm maçları bul (Ev veya Deplasman)
    tum_maclar = df[(df['HomeTeam'] == takim_adi) | (df['AwayTeam'] == takim_adi)].sort_values('Date').tail(mac_sayisi)
    
    if len(tum_maclar) == 0:
        return 0.0, 0.0, 0, 0.0

    atilan = 0
    yenilen = 0
    sut = 0
    puan = 0 # Son 5 maçı ayrıca hesaplayacağız ama burada dursun
    
    for _, row in tum_maclar.iterrows():
        if row['HomeTeam'] == takim_adi:
            atilan += row['FTHG']
            yenilen += row['FTAG']
            sut += row['HST'] if not pd.isna(row['HST']) else 0
        else:
            atilan += row['FTAG']
            yenilen += row['FTHG']
            sut += row['AST'] if not pd.isna(row['AST']) else 0
            
    avg_atilan = atilan / len(tum_maclar)
    avg_yenilen = yenilen / len(tum_maclar)
    avg_sut = sut / len(tum_maclar)
    
    # Form Hesabı (Son 5 Maç - Genel)
    son_5 = tum_maclar.tail(5)
    form_puan = 0
    for _, row in son_5.iterrows():
        if row['HomeTeam'] == takim_adi:
            p = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        else:
            p = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        form_puan += p
        
    return avg_atilan, avg_yenilen, form_puan, avg_sut

def deger_analizi_yap(yuzde, oran, tur):
    if oran <= 1.0: return ""
    deger = (yuzde / 100) * oran
    if deger >= 1.10: return f"🔥 ({deger:.2f})"
    elif deger >= 1.01: return f"✅ ({deger:.2f})"
    else: return f"❌ ({deger:.2f})"

def yuzde_goster(dataframe, baslik, renk="blue", oranlar=None):
    count = len(dataframe)
    if count == 0:
        st.warning(f"{baslik} için kriterlere uygun maç bulunamadı.")
        return

    # --- HESAPLAMALAR ---
    ms1 = len(dataframe[dataframe['FTR'] == 'H']) / count * 100
    ms0 = len(dataframe[dataframe['FTR'] == 'D']) / count * 100
    ms2 = len(dataframe[dataframe['FTR'] == 'A']) / count * 100
    kg_var = dataframe['KG_Var'].mean() * 100
    ust25 = dataframe['Over25'].mean() * 100
    
    iy1 = len(dataframe[dataframe['HTR'] == 'H']) / count * 100 if 'HTR' in dataframe.columns else 0
    iy0 = len(dataframe[dataframe['HTR'] == 'D']) / count * 100 if 'HTR' in dataframe.columns else 0
    iy2 = len(dataframe[dataframe['HTR'] == 'A']) / count * 100 if 'HTR' in dataframe.columns else 0
    iy_ust05 = dataframe['HT_Over05'].mean() * 100 if 'HT_Over05' in dataframe.columns else 0
    iy_ust15 = dataframe['HT_Over15'].mean() * 100 if 'HT_Over15' in dataframe.columns else 0
        
    ust15 = dataframe['Over15'].mean() * 100
    home05 = dataframe['Home_Over05'].mean() * 100
    home15 = dataframe['Home_Over15'].mean() * 100
    away05 = dataframe['Away_Over05'].mean() * 100
    away15 = dataframe['Away_Over15'].mean() * 100

    # Şut Ortalaması
    avg_home_sot = dataframe['HST'].mean() if 'HST' in dataframe.columns else 0
    avg_away_sot = dataframe['AST'].mean() if 'AST' in dataframe.columns else 0

    # KORNER HESAPLAMALARI
    has_corner = 'TotalCorners' in dataframe.columns
    if has_corner:
        avg_corner = dataframe['TotalCorners'].mean()
        c85_ust = dataframe['Corn_Over85'].mean() * 100
        c95_ust = dataframe['Corn_Over95'].mean() * 100
        c105_ust = dataframe['Corn_Over105'].mean() * 100
        c85_alt = 100 - c85_ust
        c95_alt = 100 - c95_ust
        c105_alt = 100 - c105_ust
    else:
        avg_corner = 0
        c85_ust = c95_ust = c105_ust = 0
        c85_alt = c95_alt = c105_alt = 0

    # --- GÖRSELLEŞTİRME ---
    st.markdown(f"### :{renk}[{baslik}] ({count} Maç)")
    
    # 1. Satır: MS
    c1, c2, c3, c4, c5 = st.columns(5)
    val_ms1 = deger_analizi_yap(ms1, oranlar.get('ms1', 0.0), 'MS 1') if oranlar else ""
    val_ms0 = deger_analizi_yap(ms0, oranlar.get('ms0', 0.0), 'MS 0') if oranlar else ""
    val_ms2 = deger_analizi_yap(ms2, oranlar.get('ms2', 0.0), 'MS 2') if oranlar else ""
    val_kg  = deger_analizi_yap(kg_var, oranlar.get('kg', 0.0), 'KG') if oranlar else ""
    val_ust = deger_analizi_yap(ust25, oranlar.get('ust', 0.0), '2.5Ü') if oranlar else ""

    c1.metric("MS 1", f"%{ms1:.1f}", delta=val_ms1)
    c2.metric("MS X", f"%{ms0:.1f}", delta=val_ms0)
    c3.metric("MS 2", f"%{ms2:.1f}", delta=val_ms2)
    c4.metric("KG VAR", f"%{kg_var:.1f}", delta=val_kg, delta_color="off")
    c5.metric("2.5 ÜST", f"%{ust25:.1f}", delta=val_ust, delta_color="off")
    
    st.write("") 
    # 2. Satır: IY
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("IY 1", f"%{iy1:.1f}")
    d2.metric("IY X", f"%{iy0:.1f}")
    d3.metric("IY 2", f"%{iy2:.1f}")
    d4.metric("IY 0.5 ÜST", f"%{iy_ust05:.1f}")
    d5.metric("IY 1.5 ÜST", f"%{iy_ust15:.1f}")

    st.write("") 
    # 3. Satır: DETAY GOL
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("1.5 ÜST", f"%{ust15:.1f}")
    e2.metric("EV > 0.5", f"%{home05:.1f}")
    e3.metric("EV > 1.5", f"%{home15:.1f}")
    e4.metric("DEP > 0.5", f"%{away05:.1f}")
    e5.metric("DEP > 1.5", f"%{away15:.1f}")

    # 4. Satır: KORNER
    if has_corner and avg_corner > 0:
        st.write("")
        st.markdown(f"**🚩 KORNER İSTATİSTİKLERİ (Ort: {avg_corner:.1f})**")
        k1, k2, k3 = st.columns(3)
        k1.metric("8.5 ÜST", f"%{c85_ust:.1f}")
        k2.metric("9.5 ÜST", f"%{c95_ust:.1f}")
        k3.metric("10.5 ÜST", f"%{c105_ust:.1f}")
        
        l1, l2, l3 = st.columns(3)
        l1.metric("8.5 ALT", f"%{c85_alt:.1f}")
        l2.metric("9.5 ALT", f"%{c95_alt:.1f}")
        l3.metric("10.5 ALT", f"%{c105_alt:.1f}")
    elif has_corner:
        st.info("⚠️ Korner verisi sütunları var ama veri boş.")
        
    if avg_home_sot > 0 or avg_away_sot > 0:
        st.caption(f"ℹ️ Bu maçlarda ortalama Ev Sahibi {avg_home_sot:.1f}, Deplasman {avg_away_sot:.1f} isabetli şut attı.")

def skor_analizi_yap(dataframe):
    if len(dataframe) == 0: return
    
    df_ft = dataframe.dropna(subset=['FTHG', 'FTAG']).copy()
    if len(df_ft) > 0:
        df_ft['Skor_Text'] = df_ft['FTHG'].astype(int).astype(str) + "-" + df_ft['FTAG'].astype(int).astype(str)
        top_skorlar = df_ft['Skor_Text'].value_counts(normalize=True).head(3) * 100
    else:
        top_skorlar = {}

    top_iy_skorlar = {}
    if 'HTHG' in dataframe.columns and 'HTAG' in dataframe.columns:
        df_ht = dataframe.dropna(subset=['HTHG', 'HTAG']).copy()
        if len(df_ht) > 0:
            df_ht['IY_Skor_Text'] = df_ht['HTHG'].astype(int).astype(str) + "-" + df_ht['HTAG'].astype(int).astype(str)
            top_iy_skorlar = df_ht['IY_Skor_Text'].value_counts(normalize=True).head(3) * 100

    st.success("📌 **EN SIK GÖRÜLEN SKORLAR**")
    
    st.markdown("##### 🏁 Maç Sonu (MS)")
    if len(top_skorlar) > 0:
        cols_ms = st.columns(3)
        for i, (skor, yuzde) in enumerate(top_skorlar.items()):
            cols_ms[i].metric(f"{i+1}. Olasılık", skor, f"%{yuzde:.1f}")
    else:
        st.write("Veri Yok")
    
    st.markdown("##### ⏱️ İlk Yarı (IY)")
    if len(top_iy_skorlar) > 0:
        cols_iy = st.columns(3)
        for i, (skor, yuzde) in enumerate(top_iy_skorlar.items()):
            cols_iy[i].metric(f"{i+1}. Olasılık", skor, f"%{yuzde:.1f}")
    else:
        st.write("IY Verisi Yok")

def ht_ft_analiz_yap(dataframe):
    if len(dataframe) == 0:
        st.info("Veri yok.")
        return

    map_dict = {'H': '1', 'D': 'X', 'A': '2'}
    if 'HTR' in dataframe.columns and 'FTR' in dataframe.columns:
        ht = dataframe['HTR'].map(map_dict)
        ft = dataframe['FTR'].map(map_dict)
        temp_df = pd.DataFrame({'HT': ht, 'FT': ft}).dropna()
        
        if len(temp_df) > 0:
            temp_df['HT_FT_Kombinasyon'] = temp_df['HT'] + "/" + temp_df['FT']
            counts = temp_df['HT_FT_Kombinasyon'].value_counts(normalize=True) * 100
            
            st.markdown("### 🎁 SÜRPRİZ İY/MS TAHMİNLERİ")
            if len(counts) > 0:
                sonuc_df = counts.reset_index()
                sonuc_df.columns = ['İY / MS', 'Olasılık (%)']
                sonuc_df['Olasılık (%)'] = sonuc_df['Olasılık (%)'].apply(lambda x: f"%{x:.1f}")
                st.table(sonuc_df)
            else:
                st.warning("Veri hesaplanamadı.")
        else:
            st.warning("Yeterli İY/MS verisi yok.")
    else:
        st.error("Veri eksik.")

# --- ARAYÜZ ---
st.title("🔮 Pro İddaa Simülasyonu (V8 - Genel Ortalama Modu)")

# SOL MENÜ
st.sidebar.header("Ayarlar")
if not os.path.exists(KLASOR_YOLU):
    st.error(f"Klasör bulunamadı: {KLASOR_YOLU}")
    st.stop()

csv_dosyalari = glob.glob(os.path.join(KLASOR_YOLU, "*.csv"))
secilen_dosya = st.sidebar.selectbox("Lig Seç:", [os.path.basename(f) for f in csv_dosyalari])
df = dosya_yukle(os.path.join(KLASOR_YOLU, secilen_dosya))

if df is None: st.stop()

# --- OTOMATİK VERİ GİRİŞİ (GENEL HESAPLAMA) ---
st.markdown("### 🏟️ Maç Analiz Girişi")
c1, c2 = st.columns(2)

takimlar = sorted(df['HomeTeam'].dropna().unique())

# --- EV SAHİBİ ---
with c1:
    st.success("🏠 Ev Sahibi")
    ev_takim = st.selectbox("Takım Seç:", takimlar, key="ev_select")
    
    if ev_takim != st.session_state.last_ev_takim:
        # GENEL HESAPLAMA (Ev + Deplasman son 10 maç)
        g_at, g_ye, g_form, g_sot = get_genel_istatistik(df, ev_takim, 10)
        
        st.session_state['ev_gol_inp'] = float(g_at)
        st.session_state['ev_yen_inp'] = float(g_ye)
        st.session_state['ev_form_inp'] = int(g_form)
        st.session_state['ev_sot_inp'] = float(g_sot)
        st.session_state.last_ev_takim = ev_takim
    
    col_e1, col_e2 = st.columns(2)
    ev_gol = col_e1.number_input("Atılan Gol Ort (Genel):", step=0.1, format="%.2f", key="ev_gol_inp", help="Takımın hem iç hem dış sahadaki son 10 maç ortalaması.")
    ev_yenilen = col_e2.number_input("Yenilen Gol Ort (Genel):", step=0.1, format="%.2f", key="ev_yen_inp")
    
    col_e3, col_e4 = st.columns(2)
    ev_sot = col_e3.number_input("İsabetli Şut (Genel):", step=0.1, format="%.2f", key="ev_sot_inp")
    ev_form = col_e4.number_input("Form Puanı (Genel):", step=1, key="ev_form_inp")

# --- DEPLASMAN ---
with c2:
    st.warning("✈️ Deplasman")
    dep_takim = st.selectbox("Takım Seç:", takimlar, index=1 if len(takimlar) > 1 else 0, key="dep_select")
    
    if dep_takim != st.session_state.last_dep_takim:
        # GENEL HESAPLAMA (Ev + Deplasman son 10 maç)
        gd_at, gd_ye, gd_form, gd_sot = get_genel_istatistik(df, dep_takim, 10)
        
        st.session_state['dep_gol_inp'] = float(gd_at)
        st.session_state['dep_yen_inp'] = float(gd_ye)
        st.session_state['dep_form_inp'] = int(gd_form)
        st.session_state['dep_sot_inp'] = float(gd_sot)
        st.session_state.last_dep_takim = dep_takim
    
    col_d1, col_d2 = st.columns(2)
    dep_gol = col_d1.number_input("Atılan Gol Ort (Genel):", step=0.1, format="%.2f", key="dep_gol_inp", help="Takımın hem iç hem dış sahadaki son 10 maç ortalaması.")
    dep_yenilen = col_d2.number_input("Yenilen Gol Ort (Genel):", step=0.1, format="%.2f", key="dep_yen_inp")
    
    col_d3, col_d4 = st.columns(2)
    dep_sot = col_d3.number_input("İsabetli Şut (Genel):", step=0.1, format="%.2f", key="dep_sot_inp")
    dep_form = col_d4.number_input("Form Puanı (Genel):", step=1, key="dep_form_inp")

st.markdown("---")

with st.expander("💰 Bahis Oranlarını Gir (Value Analizi İçin)", expanded=False):
    col_o1, col_o2, col_o3, col_o4, col_o5 = st.columns(5)
    oran_ms1 = col_o1.number_input("MS 1", value=0.0, step=0.05)
    oran_ms0 = col_o2.number_input("MS 0", value=0.0, step=0.05)
    oran_ms2 = col_o3.number_input("MS 2", value=0.0, step=0.05)
    oran_kg  = col_o4.number_input("KG Var", value=0.0, step=0.05)
    oran_ust = col_o5.number_input("2.5 Üst", value=0.0, step=0.05)

girilen_oranlar = {'ms1': oran_ms1, 'ms0': oran_ms0, 'ms2': oran_ms2, 'kg': oran_kg, 'ust': oran_ust}

col_opt1, col_opt2 = st.columns([3, 1])
with col_opt1:
    tam_eslesme = st.checkbox("🎯 Sadece Birebir (Tam) Eşleşenleri Göster")

# --- BUTONLAR ---
col_btn1, col_btn2 = st.columns(2)
run_single = col_btn1.button("🚀 SEÇİLİ LİGİ ANALİZ ET", type="primary", use_container_width=True)
run_global = col_btn2.button("🌍 TÜM LİGLERİ TARA (GLOBAL)", type="secondary", use_container_width=True)

# ANALİZ MANTIĞI
if run_single or run_global:
    st.markdown("---")
    
    if tam_eslesme:
        TOLERANS_GOL = 0.05
        TOLERANS_SOT = 0.05
        TOLERANS_FORM = 0.1
        mod_mesaji = " (Tam Eşleşme)"
    else:
        TOLERANS_GOL = 0.10
        TOLERANS_SOT = 0.5
        TOLERANS_FORM = 1.0
        mod_mesaji = ""

    # --- VERİ HAZIRLIĞI ---
    if run_single:
        hedef_dfs = [df] 
        analiz_baslik = f"Analiz Sonuçları: {secilen_dosya}"
    else:
        hedef_dfs = []
        progress_text = "Tüm ligler taranıyor..."
        my_bar = st.progress(0, text=progress_text)
        total_files = len(csv_dosyalari)
        
        for i, f_path in enumerate(csv_dosyalari):
            temp_df = dosya_yukle(f_path)
            if temp_df is not None:
                temp_df['Lig'] = os.path.basename(f_path)
                hedef_dfs.append(temp_df)
            my_bar.progress((i + 1) / total_files, text=f"Taranıyor: {os.path.basename(f_path)}")
        my_bar.empty()
        analiz_baslik = "🌍 GLOBAL ANALİZ SONUÇLARI (Tüm Ligler)"

    # --- FİLTRELEME ---
    toplanan_maclar_gol = []
    toplanan_maclar_form = []

    for curr_df in hedef_dfs:
        kosul_ev_atilan = curr_df['Home_Scored_Avg'].between(ev_gol - TOLERANS_GOL, ev_gol + TOLERANS_GOL) if ev_gol > 0 else True
        kosul_dep_atilan = curr_df['Away_Scored_Avg'].between(dep_gol - TOLERANS_GOL, dep_gol + TOLERANS_GOL) if dep_gol > 0 else True
        kosul_ev_yenilen = curr_df['Home_Conceded_Avg'].between(ev_yenilen - TOLERANS_GOL, ev_yenilen + TOLERANS_GOL) if ev_yenilen > 0 else True
        kosul_dep_yenilen = curr_df['Away_Conceded_Avg'].between(dep_yenilen - TOLERANS_GOL, dep_yenilen + TOLERANS_GOL) if dep_yenilen > 0 else True
        
        if ev_sot > 0 and 'Home_SOT_Avg' in curr_df.columns:
            kosul_ev_sot = curr_df['Home_SOT_Avg'].between(ev_sot - TOLERANS_SOT, ev_sot + TOLERANS_SOT)
        else: kosul_ev_sot = True
            
        if dep_sot > 0 and 'Away_SOT_Avg' in curr_df.columns:
            kosul_dep_sot = curr_df['Away_SOT_Avg'].between(dep_sot - TOLERANS_SOT, dep_sot + TOLERANS_SOT)
        else: kosul_dep_sot = True

        filt_gol = curr_df[kosul_ev_atilan & kosul_dep_atilan & kosul_ev_yenilen & kosul_dep_yenilen & kosul_ev_sot & kosul_dep_sot]
        if len(filt_gol) > 0: toplanan_maclar_gol.append(filt_gol)

        filt_form = curr_df[
            (curr_df['Home_Form_5'].between(ev_form - TOLERANS_FORM, ev_form + TOLERANS_FORM)) &
            (curr_df['Away_Form_5'].between(dep_form - TOLERANS_FORM, dep_form + TOLERANS_FORM))
        ]
        if len(filt_form) > 0: toplanan_maclar_form.append(filt_form)
    
    sim_gol = pd.concat(toplanan_maclar_gol) if toplanan_maclar_gol else pd.DataFrame()
    sim_form = pd.concat(toplanan_maclar_form) if toplanan_maclar_form else pd.DataFrame()

    # --- RAPORLAMA ---
    st.header(analiz_baslik)
    tab1, tab2, tab3, tab4 = st.tabs([f"📊 ANALİZ RAPORU", "⚽ LİSTE (GOL & ŞUT)", "📈 LİSTE (FORM)", "🎁 SÜRPRİZ (İY/MS)"])
    
    with tab1:
        if run_single:
            h2h = df[((df['HomeTeam'] == ev_takim) & (df['AwayTeam'] == dep_takim)) |
                     ((df['HomeTeam'] == dep_takim) & (df['AwayTeam'] == ev_takim))]
            if len(h2h) > 0:
                yuzde_goster(h2h, f"⚔️ H2H ({ev_takim}-{dep_takim})", "red")
            st.markdown("---")

        yuzde_goster(sim_gol, f"1️⃣ İstatistiksel Benzerlik (Gol & Şut){mod_mesaji}", "blue", girilen_oranlar)
        if len(sim_gol) > 0: skor_analizi_yap(sim_gol)
        
        st.markdown("---")
        yuzde_goster(sim_form, f"2️⃣ Form Benzerliği{mod_mesaji}", "green", girilen_oranlar)
        if len(sim_form) > 0: skor_analizi_yap(sim_form)

    with tab2:
        st.write(f"### İstatistiksel Benzerlik Listesi {mod_mesaji}")
        if len(sim_gol) > 0:
            cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG']
            if 'Lig' in sim_gol.columns: cols.insert(0, 'Lig')
            if 'TotalCorners' in sim_gol.columns: cols.append('TotalCorners')
            if 'Home_SOT_Avg' in sim_gol.columns: cols.append('Home_SOT_Avg')
            if 'Away_SOT_Avg' in sim_gol.columns: cols.append('Away_SOT_Avg')
            
            st.dataframe(sim_gol[cols].sort_values('Date', ascending=False), use_container_width=True)
        else:
            st.info("Kriterlere uygun maç bulunamadı.")

    with tab3:
        st.write(f"### Form Benzerliği Listesi {mod_mesaji}")
        if len(sim_form) > 0:
            cols_f = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','Home_Form_5','Away_Form_5']
            if 'Lig' in sim_form.columns: cols_f.insert(0, 'Lig')
            
            st.dataframe(sim_form[cols_f].sort_values('Date', ascending=False), use_container_width=True)
        else:
            st.info("Kriterlere uygun maç bulunamadı.")
    
    with tab4:
        st.write(f"### 🎁 İY/MS Sürpriz Listesi")
        if len(sim_gol) > 0:
            ht_ft_analiz_yap(sim_gol)
        else:
            st.info("Analiz için yeterli benzer maç bulunamadı.")