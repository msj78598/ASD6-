# ==============================================
# الخطوة الأولى: إعداد البيئة وتحميل المكتبات
# ==============================================
# الشرح:
# - هنا نقوم باستيراد المكتبات التي نحتاجها في المشروع.
# - الهدف: توفير الأدوات اللازمة لتحليل البيانات، بناء النموذج، وواجهة التفاعل.
# المكتبات:
# 1. **streamlit**: لإنشاء واجهة تفاعلية لرفع الملفات وعرض النتائج.
# 2. **pandas**: لتحليل البيانات في شكل جداول (DataFrame).
# 3. **IsolationForest**: خوارزمية لاكتشاف الشذوذ (أنماط غير طبيعية).
# 4. **train_test_split**: لتقسيم البيانات إلى مجموعات تدريب واختبار.
# 5. **SMOTE**: لمعالجة عدم التوازن في البيانات عن طريق توليد عينات اصطناعية.
# 6. **XGBoost**: خوارزمية تصنيف تعتمد على أشجار القرار وتعزيز التدرج.
# 7. **joblib**: لحفظ واسترجاع النماذج المدربة.
# 8. **os**: لإدارة الملفات والمجلدات.
# 9. **BytesIO**: لإنشاء ملفات Excel قابلة للتنزيل مباشرة.

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
from io import BytesIO

# ==============================================
# الخطوة الثانية: إعداد مسارات الملفات
# ==============================================
# الشرح:
# - تحديد مسارات حفظ الملفات مثل:
#   1. موقع النموذج المدرب.
#   2. ملف قالب البيانات المستخدم لإجراء التحليل.
#   3. ملف لحفظ مؤشرات الأداء للنموذج.
# - الهدف: ضمان سهولة الوصول للملفات وتنظيم المشروع.

model_folder = 'C:\\asd6'  # المجلد الرئيسي للمشروع
model_path = os.path.join(model_folder, 'ASD6_XGBoost.pkl')  # مسار حفظ النموذج
data_frame_template_path = 'The data frame file to be analyzed.xlsx'  # قالب البيانات
metrics_file_path = os.path.join(model_folder, 'metrics.txt')  # ملف تخزين مؤشرات الأداء

# ==============================================
# الخطوة الثالثة: دالة لتسجيل مؤشرات الأداء
# ==============================================
# الشرح:
# - الهدف: قياس أداء النموذج من خلال:
#   1. **الدقة (Accuracy)**: النسبة المئوية للتنبؤات الصحيحة من إجمالي التنبؤات.
#   2. **الدقة النوعية (Precision)**: النسبة المئوية للحالات الصحيحة المكتشفة من الحالات المصنفة إيجابيًا.
#   3. **الاسترجاع (Recall)**: نسبة الحالات المكتشفة من إجمالي الحالات الحقيقية.
#   4. **F1 Score**: مقياس يوازن بين الدقة النوعية والاسترجاع.
# - يتم حفظ النتائج في ملف نصي لسهولة الرجوع إليها.

def save_metrics_to_file(accuracy, precision, recall, f1):
    try:
        with open(metrics_file_path, 'w', encoding='utf-8') as file:
            file.write("### نتائج مؤشرات الأداء للنموذج:\n")
            file.write(f"الدقة (Accuracy): {accuracy:.2f}\n")
            file.write(f"الدقة النوعية (Precision): {precision:.2f}\n")
            file.write(f"الاسترجاع (Recall): {recall:.2f}\n")
            file.write(f"F1 Score: {f1:.2f}\n")
        st.success(f"تم حفظ نتائج الأداء في {metrics_file_path}")
    except Exception as e:
        st.error(f"حدث خطأ أثناء حفظ مؤشرات الأداء: {str(e)}")

# ==============================================
# الخطوة الرابعة: دالة لتدريب النموذج
# ==============================================
#النظام يُعد مزيجًا 
# من XGBoost 
# كخوارزمية تصنيف
# وSMOTE
# كأداة معالجة بيانات لضمان أداء عالي ودقة في اكتشاف حالات الفاقد

# - الهدف:من النظام هو بناء نموذج لتحديد حالات الفاقد.
# - الخوارزميات المستخدمة:
#XGBoost (Extreme Gradient Boosting):
# الدور: خوارزمية التصنيف الرئيسية في النظام

# الوصف: هي خوارزمية تعلم آلي تُستخدم لتصنيف البيانات
# تعتمد على تعزيز التدرج وأشجار القرار لتحديد الحالات (فاقد أو غير فاقد).
# **SMOTE**الغرض: تدريب النموذج وإجراء التنبؤات بناءً على البيانات المتوازنة الناتجة عن 

# SMOTE (Synthetic Minority Oversampling Technique):
#الدور: معالجة البيانات غير المتوازنة قبل تدريب النموذج

#الوصف:هي تقنية لتحسين توزيع البيانات 
# تقوم بإنشاء عينات صناعية للحالات الأقل تمثيلًا (مثل حالات الفاقد)، 
# مما يساعد على موازنة البيانات قبل تدريب النموذج

#الغرض: إعداد البيانات بشكل متوازن لتدريب النموذج دون تحيز

# - المعلمات المستخدمة في XGBoost:
#   - **n_estimators**: عدد أشجار القرار (كلما زاد العدد، كان النموذج أكثر دقة ولكنه أبطأ).
#   - **max_depth**: العمق الأقصى للشجرة (يساعد في التحكم في التعقيد).
#   - **learning_rate**: معدل التعلم (سرعة تحسين النموذج).

def train_and_save_model():
    try:
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        # تحميل بيانات التدريب
        file_path = r'final_classified_loss_with_reasons_60_percent_ordered.xlsx'
        data = pd.read_excel(file_path)

        # إعداد الميزات والهدف
        #X: يحتوي على البيانات المدخلة (Input Data) التي ستتعلم منها الخوارزمية.
        #y: يحتوي على القيم المستهدفة (Target Labels) التي ستساعد النموذج في التعرف على الحالات.
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        y = data['Loss_Status'].apply(lambda x: 1 if x == 'Loss' else 0)

        # معالجة توازن البيانات باستخدام SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # تقسيم البيانات إلى تدريب واختبار
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # تدريب النموذج
        model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # تقييم النموذج
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # حفظ النموذج ومؤشرات الأداء
        joblib.dump(model, model_path)
        save_metrics_to_file(accuracy, precision, recall, f1)

        # عرض مؤشرات الأداء
        st.write("### نتائج مؤشرات الأداء:")
        st.write(f"**الدقة (Accuracy)**: {accuracy:.2f}")
        st.write(f"**الدقة النوعية (Precision)**: {precision:.2f}")
        st.write(f"**الاسترجاع (Recall)**: {recall:.2f}")
        st.write(f"**F1 Score**: {f1:.2f}")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تدريب النموذج: {str(e)}")

# ==============================================
# الخطوة الخامسة: التحقق من وجود نموذج مدرب مسبقًا
# ==============================================
# الشرح:
# - الهدف: تقليل وقت التشغيل عن طريق التحقق من وجود النموذج.
# - إذا لم يكن النموذج موجودًا، يتم تدريبه تلقائيًا.

if not os.path.exists(model_path):
    train_and_save_model()
else:
    st.write("يتم استخدام النموذج المدرب مسبقا لاكتشاف حالات الفاقد ")

# ==============================================
# الخطوة السادسة: دالة لإضافة أسباب الفاقد
# ==============================================
# الشرح:
# - الهدف: تحديد السبب لكل حالة فاقد بناءً على القيم المدخلة.

# دالة لإضافة أسباب الفاقد بناءً على شروط التحليل
def add_loss_reason(row):
    if row['V1'] == 0 and row['A1'] > 0:
        return 'فاقد بسبب جهد صفر مع تيار على V1'
    elif row['V2'] == 0 and row['A2'] > 0:
        return 'فاقد بسبب جهد صفر مع تيار على V2'
    elif row['V3'] == 0 and row['A3'] > 0:
        return 'فاقد بسبب جهد صفر مع تيار على V3'
    elif row['V1'] == 0 and row['A1'] == 0 and abs(row['A2'] - row['A3']) > 0.6 * max(row['A2'], row['A3']):
        return 'فاقد بسبب جهد وتيار صفر على V1 مع فرق كبير بين A2 وA3'
    elif row['V2'] == 0 and row['A2'] == 0 and abs(row['A1'] - row['A3']) > 0.6 * max(row['A1'], row['A3']):
        return 'فاقد بسبب جهد وتيار صفر على V2 مع فرق كبير بين A1 وA3'
    elif row['V3'] == 0 and row['A3'] == 0 and abs(row['A1'] - row['A2']) > 0.6 * max(row['A1'], row['A2']):
        return 'فاقد بسبب جهد وتيار صفر على V3 مع فرق كبير بين A1 وA2'
    elif row['V1'] < 10 and row['A1'] > 0:
        return 'فاقد بسبب جهد منخفض مع تيار على V1'
    elif row['V2'] < 10 and row['A2'] > 0:
        return 'فاقد بسبب جهد منخفض مع تيار على V2'
    elif row['V3'] < 10 and row['A3'] > 0:
        return 'فاقد بسبب جهد منخفض مع تيار على V3'
    elif abs(row['A1'] - row['A2']) > 0.6 * max(row['A1'], row['A2']) and row['A3'] == 0:
        return 'فاقد بسبب فرق تيار كبير بين A1 و A2 مع صفر تيار على A3'
    else:
        return 'اسباب اخرى لحالات فاقد محتمله'

# ==============================================
# الخطوة السابعة: دالة لتحليل البيانات
# ==============================================
# الشرح:
# - الهدف: استخدام النموذج لتحليل البيانات وتحديد حالات الفاقد.

def analyze_data(data):
    try:
        model = joblib.load(model_path)
        X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
        data['Predicted_Loss'] = model.predict(X)
        data['Reason'] = data.apply(add_loss_reason, axis=1)
        loss_data = data[data['Predicted_Loss'] == 1]
        st.write(f"### عدد حالات الفاقد المكتشفة: {len(loss_data)}")
        st.dataframe(loss_data)
    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

# ==============================================
# الخطوة الثامنة: إضافة قالب البيانات في واجهة المستخدم
# ==============================================
# الشرح:
# - الهدف: توفير قالب بيانات جاهز للمستخدم لتنظيم البيانات بالشكل المطلوب.
# - يتم عرض زر لتحميل ملف القالب المخزن مسبقًا في المسار المحدد.
# ==============================================

st.header("تنزيل قالب البيانات")
st.write("استخدم هذا القالب لتنسيق بياناتك بشكل صحيح قبل رفعها للتحليل.")
try:
    with open(data_frame_template_path, 'rb') as template_file:
        template_data = template_file.read()
    st.download_button(
        label="تحميل قالب البيانات",
        data=template_data,
        file_name="The_data_frame_file_to_be_analyzed.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except Exception as e:
    st.error(f"تعذر تحميل قالب البيانات: {str(e)}")


# ==============================================
# الخطوة التاسعة: واجهة المستخدم باستخدام Streamlit
# ==============================================
# الشرح:
# - الهدف: إنشاء واجهة تفاعلية لتحليل البيانات وعرض النتائج.

st.title("تحليل الاحمال للتبأ بحالات الفاقد")
uploaded_file = st.file_uploader("قم برفع ملف البيانات", type=["xlsx"])
if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        analyze_data(data)
    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {str(e)}")
