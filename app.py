from pathlib import Path
import hashlib
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from src.batch import (DUPLICATE, INVALID, NO_ASPECT, PREDICTED, analyze_table, csv_bytes,
                       read_table)
from src.data import ASPECT_VI, SENTIMENT_VI, valid_comment
from src.models import load_models, predict_ABSA

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='ABSA điện thoại', page_icon='📱', layout='wide')


@st.cache_resource
def get_models():
    return load_models(ROOT / 'models')


def show_dashboard(audit, output):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Dòng đầu vào', len(audit))
    c2.metric('Đã phân tích', int(audit['trang_thai'].isin([PREDICTED, NO_ASPECT]).sum()))
    c3.metric('Bỏ qua do trùng', int(audit['trang_thai'].eq(DUPLICATE).sum()))
    c4.metric('Bỏ qua do không hợp lệ', int(audit['trang_thai'].eq(INVALID).sum()))
    st.caption(f"{audit['trang_thai'].eq(NO_ASPECT).sum()} bình luận hợp lệ không có khía cạnh "
               "vượt ngưỡng. Dashboard tính trên các cặp được dự đoán.")
    pairs = output.dropna(subset=['aspect'])
    if pairs.empty:
        st.info('Chưa có cặp khía cạnh và cảm xúc để vẽ biểu đồ.')
    else:
        left, right = st.columns(2)
        left.subheader('Lượt nhắc theo khía cạnh')
        left.bar_chart(pairs['khia_canh'].value_counts().rename('Lượt nhắc'), horizontal=True)
        right.subheader('Tỷ lệ cảm xúc')
        shares = pairs['cam_xuc'].value_counts(normalize=True).mul(100).rename('Tỷ lệ %')
        right.bar_chart(shares)
        by_aspect = pd.crosstab(pairs['khia_canh'], pairs['cam_xuc'], normalize='index').mul(100)
        with st.expander('Tỷ lệ cảm xúc trong từng khía cạnh (%)'):
            st.dataframe(by_aspect.reindex(columns=list(SENTIMENT_VI.values()), fill_value=0).round(1))
    st.subheader('Chi tiết kết quả')
    aspects = sorted(pairs['khia_canh'].unique())
    aspect_filter = st.selectbox('Lọc khía cạnh', [None, *aspects],
                                 format_func=lambda x: x or 'Tất cả khía cạnh')
    products = sorted({str(v) for v in pairs['san_pham'] if pd.notna(v) and str(v).strip()})
    product = st.selectbox('Lọc sản phẩm', [None, *products],
                           format_func=lambda x: x or 'Tất cả sản phẩm') if products else None
    negative = st.checkbox('Chỉ hiện bình luận tiêu cực')
    filtered = output
    if aspect_filter:
        filtered = filtered.loc[filtered['khia_canh'].eq(aspect_filter)]
    if product:
        filtered = filtered.loc[filtered['san_pham'].astype(str).eq(product)]
    if negative:
        filtered = filtered.loc[filtered['sentiment'].eq('Negative')]
    st.dataframe(filtered, hide_index=True, width='stretch')
    st.download_button('Tải CSV kết quả đầy đủ', csv_bytes(output), 'absa_ket_qua.csv',
                       'text/csv', on_click='ignore')
    if aspect_filter or product or negative:
        st.download_button('Tải CSV theo bộ lọc', csv_bytes(filtered), 'absa_da_loc.csv',
                           'text/csv', on_click='ignore')


st.title('Phân tích đánh giá điện thoại')
st.caption('Mô hình UIT-ViSFD nhận diện khía cạnh và cảm xúc trong bình luận tiếng Việt.')
try:
    aspect, sentiment = get_models()
except (OSError, ValueError, ImportError) as err:
    st.error(f'Không tải được mô hình. Kiểm tra thư mục models và requirements.txt. {err}')
    st.stop()
if sentiment['backend'] == 'phobert':
    sentiment = {**sentiment, 'checkpoint': ROOT / 'models' / sentiment['checkpoint']}
single, batch = st.tabs(['Một bình luận', 'Phân tích tệp'])
with single:
    with st.form('single_review'):
        review = st.text_area('Bình luận tiếng Việt', placeholder='Camera đẹp nhưng pin yếu.',
                              max_chars=4000)
        submitted = st.form_submit_button('Phân tích bình luận', type='primary')
    if submitted:
        if not valid_comment(review):
            st.warning('Nhập bình luận có ít nhất một chữ cái.')
        else:
            pairs = predict_ABSA(review, aspect, sentiment)[0]
            if pairs:
                st.dataframe(pd.DataFrame([(ASPECT_VI[a], SENTIMENT_VI[s], a, s)
                                           for a, s in pairs.items()],
                                          columns=['Khía cạnh', 'Cảm xúc', 'Mã', 'Nhãn']),
                             hide_index=True, width='stretch')
            else:
                st.info('Không phát hiện khía cạnh vượt ngưỡng.')
with batch:
    st.caption('Tệp CSV hoặc XLSX, tối đa 10.000 dòng. Với XLSX, ứng dụng đọc sheet đầu tiên.')
    demo = st.checkbox('Dùng tệp mẫu của dự án')
    uploaded = st.file_uploader('Tải tệp đánh giá', type=['csv', 'xlsx'])
    source = (('danh_gia_mau.csv', (ROOT / 'samples/danh_gia_mau.csv').read_bytes()) if demo
              else (uploaded.name, uploaded.getvalue()) if uploaded else None)
    if source:
        name, raw = source
        try:
            frame = read_table(name, raw)
            if frame.empty:
                raise ValueError('Tệp không có dòng dữ liệu.')
        except (ValueError, OSError, ImportError, BadZipFile, pd.errors.ParserError) as err:
            st.error(f'Không đọc được tệp. {err}')
        else:
            columns = frame.columns.tolist()
            st.write(f'**{name}** · {len(frame):,} dòng')
            selected = st.columns(3)
            comment = selected[0].selectbox('Cột bình luận', columns,
                                             index=columns.index('comment') if 'comment' in columns else 0)
            metadata = [None, *(c for c in columns if c != comment)]
            product = selected[1].selectbox('Cột sản phẩm (tùy chọn)', metadata,
                                             index=metadata.index('product') if 'product' in metadata else 0)
            times = [c for c in metadata if c != product or c is None]
            time = selected[2].selectbox('Cột thời gian (tùy chọn)', times,
                                         index=times.index('date') if 'date' in times else 0)
            signature = (hashlib.sha256(raw).hexdigest(), comment, product, time)
            if st.button('Phân tích tệp', type='primary'):
                try:
                    with st.spinner('Đang dự đoán theo lô…'):
                        results = analyze_table(frame, comment, product, time, aspect, sentiment)
                    st.session_state['batch_result'] = (signature, results)
                except ValueError as err:
                    st.error(str(err))
            stored = st.session_state.get('batch_result')
            if stored and stored[0] == signature:
                show_dashboard(*stored[1])
