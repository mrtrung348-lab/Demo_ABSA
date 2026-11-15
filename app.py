from hashlib import sha256
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile

import pandas as pd
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException

from src.batch import analyze_table, read_table
from src.dashboard import show_dashboard
from src.data import valid_comment
from src.models import load_models

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / 'samples' / 'danh_gia_mau.csv'
EXAMPLE = 'Camera chụp ảnh đẹp, nhưng pin tụt nhanh. Màn hình sáng và rõ nét.'
MAX_ROWS = 10_000

st.set_page_config(page_title='ABSA Studio | Đánh giá điện thoại', page_icon='📱', layout='wide')


@st.cache_resource
def get_models():
    aspect, sentiment = load_models(ROOT / 'models')
    if sentiment.get('backend') == 'phobert':
        sentiment = {**sentiment, 'checkpoint': ROOT / 'models' / sentiment['checkpoint']}
    return aspect, sentiment


def load_input(name, content):
    frame = read_table(name, content)
    if frame.empty:
        raise ValueError('Tệp chỉ có tiêu đề hoặc chưa có dữ liệu.')
    if len(frame) > MAX_ROWS:
        raise ValueError('Tệp vượt giới hạn 10.000 dòng.')
    return frame


def analyze(frame, comment, product=None, time=None):
    with st.spinner('Đang nhận diện khía cạnh và cảm xúc…'):
        return analyze_table(frame, comment, product, time, *get_models())


def single_view():
    form_col, help_col = st.columns([2.25, 1], gap='medium')
    with form_col, st.container(border=True, key='manual_panel'):
        st.subheader('Nhập bình luận')
        with st.form('manual_form', border=False):
            review = st.text_area(
                'Bình luận tiếng Việt', key='manual_review', max_chars=4000, height=132,
                placeholder='Ví dụ: Camera chụp ảnh đẹp, nhưng pin tụt nhanh.',
            )
            submitted = st.form_submit_button('Phân tích bình luận', type='primary')
    with help_col, st.container(border=True, key='example_panel'):
        st.markdown('##### Bình luận mẫu')
        st.markdown('“Camera chụp ảnh đẹp, nhưng pin tụt nhanh. Màn hình sáng và rõ nét.”')
        st.button(
            'Dùng bình luận mẫu',
            on_click=set_example,
        )

    if submitted:
        st.session_state.pop('manual_output', None)
        if not valid_comment(review):
            st.warning('Nhập bình luận có ít nhất một chữ cái.')
        else:
            frame = pd.DataFrame({'comment': [review]})
            try:
                st.session_state.manual_output = (review, analyze(frame, 'comment'))
            except (OSError, ValueError, ImportError) as error:
                st.error(f'Không thể phân tích bình luận. {error}')

    saved = st.session_state.get('manual_output')
    if saved and saved[0] == review:
        show_dashboard(*saved[1], key='manual')


def set_example():
    st.session_state.manual_review = EXAMPLE


def column_picker(label, options, preferred, key):
    return st.selectbox(
        label, options,
        index=options.index(preferred) if preferred in options else 0,
        format_func=lambda label: 'Không chọn' if label is None else str(label),
        key=key,
    )


def batch_view():
    upload_col, help_col = st.columns([2.25, 1], gap='medium')
    with upload_col, st.container(border=True, key='upload_panel'):
        st.subheader('Tải tệp bình luận')
        source = st.radio(
            'Nguồn dữ liệu', ['Tải tệp của bạn', 'Dùng tệp mẫu'], horizontal=True,
        )
        if source == 'Dùng tệp mẫu':
            name, content = SAMPLE.name, SAMPLE.read_bytes()
        else:
            uploaded = st.file_uploader(
                'Tệp đánh giá (CSV hoặc XLSX)', type=['csv', 'xlsx'],
                key='review_file',
            )
            name, content = (uploaded.name, uploaded.getvalue()) if uploaded else (None, None)

    with help_col, st.container(border=True, key='file_help'):
        st.markdown('##### Định dạng tệp')
        st.markdown('Mỗi dòng là một bình luận. Với XLSX, ứng dụng đọc trang tính đầu tiên.')
        st.caption('Tối đa 10 MB và 10.000 dòng mỗi tệp.')
        st.download_button(
            'Tải CSV mẫu', SAMPLE.read_bytes(), SAMPLE.name, 'text/csv', on_click='ignore',
        )

    if content is None:
        return
    digest = sha256(content).hexdigest()
    source_key = (Path(name).suffix.lower(), digest)
    parsed = st.session_state.get('parsed_file')
    try:
        if parsed and parsed[0] == source_key:
            frame = parsed[1]
        else:
            frame = load_input(name, content)
            st.session_state.parsed_file = (source_key, frame)
    except (ValueError, OSError, ImportError, BadZipFile, InvalidFileException, ParseError, KeyError) as error:
        st.error(f'Không đọc được tệp. {error}')
        return

    with st.container(border=True, key='mapping_panel'):
        st.subheader('Chọn cột dữ liệu')
        st.text(f'{name} · {len(frame):,} dòng')
        columns = frame.columns.tolist()
        review_col, product_col, date_col = st.columns(3)
        with review_col:
            comment = column_picker('Bình luận', columns, 'comment', f'comment-{digest}')
        metadata = [None, *(col for col in columns if col != comment)]
        with product_col:
            product = column_picker('Sản phẩm (tùy chọn)', metadata, 'product', f'product-{digest}-{comment}')
        remaining = [col for col in metadata if col is None or col != product]
        with date_col:
            time = column_picker(
                'Thời gian (tùy chọn)', remaining, 'date', f'time-{digest}-{comment}-{product}',
            )
        with st.expander('Xem trước dữ liệu'):
            st.dataframe(frame.head(5), hide_index=True, width='stretch')
        signature = (name, digest, comment, product, time)
        if st.button('Phân tích tệp', type='primary'):
            st.session_state.pop('batch_output', None)
            try:
                st.session_state.batch_output = (signature, analyze(frame, comment, product, time))
            except (OSError, ValueError, ImportError) as error:
                st.error(f'Không thể phân tích tệp. {error}')

    saved = st.session_state.get('batch_output')
    if saved and saved[0] == signature:
        show_dashboard(*saved[1], key='batch')


st.markdown(f"<style>{(ROOT / 'assets' / 'theme.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)
st.markdown(
    """
    <div class="absa-hero">
      <div class="absa-brand"><span aria-hidden="true"></span> ABSA STUDIO</div>
      <h1>Phân tích đánh giá<br class="hero-break"> điện thoại</h1>
      <p>Xem khách hàng khen hay chê điều gì trong bình luận tiếng Việt, từ pin và camera đến giá bán.</p>
      <div class="hero-bars" aria-hidden="true">
        <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
manual, batch = st.tabs(['Một bình luận', 'Phân tích tệp'])
with manual:
    single_view()
with batch:
    batch_view()

st.caption('Mô hình ABSA huấn luyện trên bộ dữ liệu đánh giá điện thoại tiếng Việt UIT-ViSFD.')
