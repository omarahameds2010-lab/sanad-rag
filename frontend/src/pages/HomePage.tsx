import type { HealthResponse } from '../types';

interface Props {
  health: HealthResponse | null;
  evalHealth: { status: string; llm_ok?: boolean } | null;
}

export default function HomePage({ health, evalHealth }: Props) {
  return (
    <div>
      <div className="card">
        <h2>سند — الأساس اللي بترتكز عليه</h2>
        <p className="desc">
          منصة تقييم عربي أولًا: إدخال مواد مرجعية، استرجاع هجين، ثم تقييم إنشائي
          واعٍ بالروبرك مع درجات لكل معيار وأدلة من إجابة الطالب.
        </p>
        <div className="grid two">
          <div>
            <div className="kv"><span>حالة الخدمة</span><span>{health ? health.status : '—'}</span></div>
            <div className="kv"><span>Qdrant</span><span>{health ? (health.qdrant_ok ? 'يعمل' : 'متعطل') : '—'}</span></div>
            <div className="kv"><span>قاعدة البيانات</span><span>{health ? (health.db_ok ? 'تعمل' : 'متعطلة') : '—'}</span></div>
            <div className="kv"><span>نموذج اللغة</span><span>{evalHealth ? (evalHealth.llm_ok ? 'حقيقي' : 'تجريبي (FakeLLM)') : '—'}</span></div>
          </div>
          <div>
            <div className="kv"><span>1. أدخل المادة</span><span>تبويب «إدخال مادة»</span></div>
            <div className="kv"><span>2. اسأل أو قيّم</span><span>«اسأل» للأسئلة · «تقييم إجابة» للتصحيح</span></div>
            <div className="kv"><span>3. راجع المعلم</span><span>اعتمد أو عدّل الدرجة في منصتك</span></div>
            <div className="kv"><span>ملاحظة</span><span>الدرجة الرسمية تبقى في الـ LMS دائمًا</span></div>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>حدود مهمة قبل الاستخدام</h2>
        <p className="desc">
          هذه الواجهة للتجربة والديمو. مفتاح API هنا للتجربة المحلية فقط —
          في الإنتاج يجب أن تكون كل calls من الـ backend (server-to-server).
        </p>
        <div className="kv"><span>أنواع الأسئلة المدعومة</span><span>ESSAY / SHORT_ANSWER</span></div>
        <div className="kv"><span>مجموع المعايير</span><span>يجب أن يساوي الدرجة العظمى</span></div>
        <div className="kv"><span>الأخطاء القابلة لإعادة المحاولة</span><span className="mono">429 / 500 / 503 / timeout</span></div>
      </div>
    </div>
  );
}
