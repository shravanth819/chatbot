"""
Agri-Mitra — RAG Co-Pilot Service
Hybrid engine: Uses LangChain + Gemini + ChromaDB when GOOGLE_API_KEY is present,
and falls back to an intelligent, curated Offline Agronomic Knowledge Engine from ICAR & FAO guides.
"""
import os
import re
import pymupdf as fitz

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings = None, None

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyMuPDFLoader
except ImportError:
    Chroma, PyMuPDFLoader = None, None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
except ImportError:
    ChatPromptTemplate, RunnablePassthrough, StrOutputParser = None, None, None

from app.core.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are Agri-Mitra Co-Pilot, an expert agronomist for Indian farmers.
Answer ONLY based on the provided context from ICAR/FAO documents.
Always cite your source using [Source: <document>] notation.
If the answer is not in the context, respond with exactly: "Data Not Available"
Never hallucinate or guess beyond the provided context.
Answer in {language}. Keep answers concise and practical for farmers.

Context:
{context}

Question: {question}
"""

GREETING_KEYWORDS = {
    "hi", "hii", "hiii", "hello", "helloo", "hey", "heyy", "namaste", "namaskara", "vanakkam", "pranam",
    "नमस्ते", "नमस्कार", "प्रणाम", "ಹಲೋ", "ನಮಸ್ಕಾರ", "kem cho", "hola"
}

THANKS_KEYWORDS = {
    "thanks", "thank you", "thx", "dhanyawad", "shukriya", "धन्यवाद", "ಶುಕ್ರಿಯಾ", "ಧನ್ಯವಾದಗಳು"
}

GREETING_RESPONSES = {
    "English": (
        "Namaste! 🌱 I am **Agri-Mitra Co-Pilot**, your AI agronomist backed by ICAR and FAO knowledge bases.\n\n"
        "Here is how I can help you today:\n"
        "• **Fertilizer Guidance**: Recommended N-P-K doses and split application schedules\n"
        "• **Soil Health**: Optimal pH ranges, salinity correction, and organic management\n"
        "• **Irrigation Advice**: Critical crop growth stages and FAO-56 ET₀ evapotranspiration\n"
        "• **Yield & Crops**: Recommended crops and achievable yield targets\n\n"
        "Feel free to ask a question or tap one of the quick prompts above!"
    ),
    "Hindi": (
        "नमस्ते! 🌱 मैं **कृषि-मित्र को-पायलट** हूँ, आपका डिजिटल कृषि विशेषज्ञ (ICAR और FAO दिशानिर्देशों पर आधारित)।\n\n"
        "मैं आपकी इन विषयों में मदद कर सकता हूँ:\n"
        "• **उर्वरक सलाह**: चावल, गेहूं, मक्का और कपास के लिए NPK खुराक और सही समय\n"
        "• **मिट्टी का स्वास्थ्य**: मिट्टी का सही pH और सुधार के उपाय\n"
        "• **सिंचाई योजना**: फसल की महत्वपूर्ण अवस्थाओं में सही समय पर पानी देना\n"
        "• **फसल चयन**: आपकी मिट्टी और मौसम के अनुसार सबसे उपयुक्त फसल\n\n"
        "कोई भी प्रश्न पूछें या ऊपर दिए गए सुझावों पर क्लिक करें!"
    ),
    "Kannada": (
        "ನಮಸ್ಕಾರ! 🌱 ನಾನು **ಕೃಷಿ-ಮಿತ್ರ ಸಹಾಯಕ (Co-Pilot)**, ನಿಮ್ಮ AI ಕೃಷಿ ಸಲಹೆಗಾರ (ICAR ಮತ್ತು FAO ಮಾರ್ಗಸೂಚಿಗಳ ಆಧಾರದ ಮೇಲೆ).\n\n"
        "ನಾನು ನಿಮಗೆ ಈ ಕೆಳಗಿನವುಗಳಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ:\n"
        "• **ರಸಗೊಬ್ಬರ ಪ್ರಮಾಣ**: ಬೆಳೆಗಳಿಗೆ ಸಾರಜನಕ, ರಂಜಕ, ಪೊಟ್ಯಾಷ್ ಸರಿಯಾದ ಪ್ರಮಾಣ\n"
        "• **ಮಣ್ಣಿನ ಆರೋಗ್ಯ**: ಮಣ್ಣಿನ ಸೂಕ್ತ pH ಮತ್ತು ನಿರ್ವಹಣೆ\n"
        "• **ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿ**: ಬೆಳೆಯ ಪ್ರಮುಖ ಹಂತಗಳಲ್ಲಿ ಸರಿಯಾದ ನೀರಾವರಿ\n"
        "• **ಬೆಳೆ ಆಯ್ಕೆ**: ನಿಮ್ಮ ಭೂಮಿಗೆ ಸೂಕ್ತವಾದ ಇಳುವರಿ ಬೆಳೆಗಳು\n\n"
        "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ!"
    ),
    "Telugu": (
        "నమస్కారం! 🌱 నేను **అగ్రి-మిత్ర కో-పైలట్**, మీ డిజిటల్ వ్యవసాయ నిపుణుడిని (ICAR మరియు FAO మార్గదర్శకాల ఆధారంగా).\n\n"
        "నేను మీకు ఈ క్రింది విషయాలలో సహాయం చేయగలను:\n"
        "• **ఎరువుల మోతాదు**: వరి, గోధుమ, మొక్కజొన్న మరియు పత్తికి NPK ఎరువుల సరైన పరిమాణం\n"
        "• **నేల ఆరోగ్యం**: నేల సరైన pH మరియు సవరణ చర్యలు\n"
        "• **నీటిపారుదల ప్రణాళిక**: పంట ముఖ్యమైన దశలలో సరైన సమయానికి నీరు ఇవ్వడం\n"
        "• **పంట ఎంపిక**: మీ నేలకు అనువైన దిగుబడి పంటలు\n\n"
        "దయచేసి మీ ప్రశ్నను అడగండి!"
    ),
    "Tamil": (
        "வணக்கம்! 🌱 நான் **அக்ரி-மித்ரா கோ-பைலட்**, உங்கள் டிஜிட்டல் வேளாண்மை ஆலோசகர் (ICAR மற்றும் FAO வழிகாட்டுதல்கள் அடிப்படையில்).\n\n"
        "நான் உங்களுக்கு பின்வரும் வழிகளில் உதவ முடியும்:\n"
        "• **உர மேலாண்மை**: நெல், கோதுமை, மக்காச்சோளம் மற்றும் பருத்திக்கு NPK உர அளவு\n"
        "• **மண் வளம்**: உகந்த pH அளவு மற்றும் திருத்த நடவடிக்கைகள்\n"
        "• **பாசன திட்டமிடல்**: பயிர்களின் முக்கியமான வளர்ச்சி நிலைகளில் சரியான பாசனம்\n"
        "• **பயிர் தேர்வு**: உங்கள் நிலத்திற்கு ஏற்ற சிறந்த பயிர்கள்\n\n"
        "தயவுசெய்து உங்கள் கேள்வியைக் கேளுங்கள்!"
    ),
    "Marathi": (
        "नमस्कार! 🌱 मी **ॲग्री-मित्र को-पायलट** आहे, तुमचा डिजिटल शेती सल्लागार (ICAR आणि FAO मार्गदर्शक तत्त्वांवर आधारित).\n\n"
        "मी तुम्हाला पुढील गोष्टींमध्ये मदत करू शकतो:\n"
        "• **खत व्यवस्थापन**: भात, गहू, मका आणि कापूस पिकांसाठी NPK खतांचे योग्य प्रमाण\n"
        "• **जमिनीचे आरोग्य**: जमिनीचा योग्य सामू (pH) आणि सुधारणा\n"
        "• **पाणी व्यवस्थापन**: पिकांच्या महत्त्वाच्या वाढीच्या अवस्थेत पाणी देणे\n"
        "• **पीक निवड**: तुमच्या जमिनीसाठी सर्वात योग्य फायदेशीर पीक\n\n"
        "कृपया तुमचा प्रश्न विचारा!"
    ),
    "Bengali": (
        "নমস্কার! 🌱 আমি **অ্যাগ্রি-মিত্র কো-পাইলট**, আপনার ডিজিটাল কৃষি বিশেষজ্ঞ (ICAR ও FAO নির্দেশিকা ভিত্তিক)।\n\n"
        "আমি আপনাকে সাহায্য করতে পারি:\n"
        "• **সার প্রয়োগের নিয়ম**: ধান, গম, ভুট্টা ও তুলার জন্য NPK সারের সঠিক মাত্রা\n"
        "• **মাটির স্বাস্থ্য**: মাটির সঠিক pH এবং লবণাক্ততা নিয়ন্ত্রণ\n"
        "• **সেচ পরিকল্পনা**: ফসলের বৃদ্ধির গুরুত্বপূর্ণ পর্যায়ে সঠিক সেচ\n"
        "• **ফসল নির্বাচন**: আপনার মাটির উপযোগী সেরা ফসল\n\n"
        "আপনার প্রশ্ন জিজ্ঞাসা করুন!"
    ),
    "Gujarati": (
        "નમસ્તે! 🌱 હું **એગ્રી-મિત્ર કો-પાયલટ** છું, તમારો ડિજિટલ કૃષિ સલાહકાર (ICAR અને FAO માર્ગદર્શિકા આધારિત).\n\n"
        "હું તમને નીચેની બાબતોમાં મદદ કરી શકું છું:\n"
        "• **ખાતર વ્યવસ્થાપન**: ડાંગર, ઘઉં, મકાઈ અને કપાસ માટે NPK ખાતરની ભલામણ\n"
        "• **જમીન આરોગ્ય**: જમીનનો સાચો pH અને સુધારણા\n"
        "• **પિયત આયોજન**: પાકના મહત્વના તબક્કે પાણી આપવું\n"
        "• **પાક પસંદગી**: તમારી જમીન માટે સૌથી યોગ્ય પાક\n\n"
        "કૃપા કરીને તમારો પ્રશ્ન પૂછો!"
    ),
    "Malayalam": (
        "നമസ്കാരം! 🌱 ഞാൻ **അഗ്രി-മിത്ര കോ-പൈലറ്റ്**, നിങ്ങളുടെ ഡിജിറ്റൽ കാർഷിക വിദഗ്ദ്ധൻ (ICAR & FAO മാർഗ്ഗനിർദ്ദേശങ്ങൾ അടിസ്ഥാനമാക്കി).\n\n"
        "ഞാൻ നിങ്ങളെ സഹായിക്കാം:\n"
        "• **വളപ്രയോഗം**: നെല്ല്, ഗോതമ്പ്, ചോളം, പരുത്തി എന്നിവയ്ക്കുള്ള NPK അളവുകൾ\n"
        "• **മണ്ണിന്റെ ആരോഗ്യം**: അനുയോജ്യമായ pH പരിപാലനം\n"
        "• **ജലസേചന ക്രമം**: നിർണ്ണായക ഘട്ടങ്ങളിലെ ജലസേചനം\n"
        "• **വിള തിരഞ്ഞെടുപ്പ്**: നിങ്ങളുടെ മണ്ണിന് അനുയോജ്യമായ മികച്ച വിളകൾ\n\n"
        "ദയവായി ചോദ്യങ്ങൾ ചോദിക്കൂ!"
    ),
}

THANKS_RESPONSES = {
    "English": "You're very welcome! Let me know if you need any more advice for your field. May your harvest be bountiful! 🌾",
    "Hindi": "आपका बहुत-बहुत स्वागत है! यदि आपको अपने खेत के लिए और कोई सलाह चाहिए तो अवश्य पूछें। आपकी फसल अच्छी हो! 🌾",
    "Kannada": "ತುಂಬಾ ಧನ್ಯವಾದಗಳು! ನಿಮ್ಮ ಕೃಷಿ ಸಂಬಂಧಿತ ಯಾವುದೇ ಪ್ರಶ್ನೆಗಳಿದ್ದರೂ ಕೇಳಬಹುದು. ಉತ್ತಮ ಬೆಳೆ ನಿಮ್ಮದಾಗಲಿ! 🌾",
    "Telugu": "చాలా ధన్యవాదాలు! మీ పంటలకు సంబంధించి ఇంకా ఏవైనా సలహాలు కావాలంటే అడగండి. మీకు మంచి దిగుబడి రావాలని కోరుకుంటున్నాము! 🌾",
    "Tamil": "மிக்க நன்றி! உங்கள் நிலத்திற்கு மேலும் ஏதேனும் ஆலோசனை தேவைப்பட்டால் கேட்கலாம். உங்கள் மகசூல் பெருக வாழ்த்துகள்! 🌾",
    "Marathi": "खूप खूप धन्यवाद! तुमच्या शेतीविषयी आणखी काही प्रश्न असल्यास नक्की विचारा. तुमचे पीक भरघोस येवो! 🌾",
    "Bengali": "আপনাকে অনেক ধন্যবাদ! আপনার জমির বিষয়ে আরও পরামর্শ লাগলে অবশ্যই জানাবেন। আপনার ফলন ভালো হোক! 🌾",
    "Gujarati": "ખૂબ ખૂબ આભાર! જો તમને તમારા ખેતર માટે કોઈ વધુ સલાહની જરૂર હોય તો જરૂર પૂછો. તમારી ઉપજ ઉત્તમ રહે! 🌾",
    "Malayalam": "വളരെ നന്ദി! നിങ്ങളുടെ കൃഷിയിടത്തെക്കുറിച്ച് കൂടുതൽ ഉപദേശങ്ങൾ വേണമെങ്കിൽ ചോദിക്കൂ. നല്ല വിളവ് ആശംസിക്കുന്നു! 🌾",
}


class KnowledgeBase:
    """
    Curated, verified ICAR and FAO agronomic knowledge for direct fast-path answering.
    """
    KNOWLEDGE = {
        "nitrogen": {
            "keywords": ["nitrogen", "urea", "नाइट्रोजन", "यूरिया", "ಸಾರಜನಕ", "నత్రజని", "நைட்ரஜன்", "नत्र", "নাইট্রোজেন", "નાઇટ્રોજન", "n dose", "fertilizer dose"],
            "English": (
                "**ICAR Recommended Nitrogen Application for Indian Crops:**\n\n"
                "• **Rice (Paddy)**: 120 kg N/ha in 3 split doses:\n"
                "  - 50% (60 kg) as basal dose at transplanting\n"
                "  - 33% (40 kg) at active tillering (21-25 days after transplanting)\n"
                "  - 17% (20 kg) at panicle initiation\n"
                "• **Wheat**: 120 kg N/ha in 3 splits (60 kg basal + 40 kg CRI stage + 20 kg at boot leaf stage)\n"
                "• **Maize**: 150 kg N/ha in 3 equal splits (at sowing, knee-high, and tasseling)\n"
                "• **Cotton**: 120 kg N/ha in 3 splits (basal, squaring, and boll development)\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Hindi": (
                "**ICAR अनुशंसित नाइट्रोजन (यूरिया) प्रबंधन:**\n\n"
                "• **चावल (धान)**: 120 किग्रा N/हेक्टेयर (3 खुराकों में):\n"
                "  - 50% (60 किग्रा) रोपाई के समय बेसल डोज\n"
                "  - 33% (40 किग्रा) कल्ले फूटते समय (20-25 दिन बाद)\n"
                "  - 17% (20 किग्रा) बाली निकलते समय\n"
                "• **गेहूं**: 120 किग्रा N/हेक्टेयर (60 किग्रा बुवाई + 40 किग्रा पहली सिंचाई CRI + 20 किग्रा गभोट अवस्था)\n"
                "• **मक्का**: 150 किग्रा N/हेक्टेयर (3 बराबर खुराकों में)\n"
                "• **कपास**: 120 किग्रा N/हेक्टेयर\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Kannada": (
                "**ICAR ಸಾರಜನಕ (Nitrogen) ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು:**\n\n"
                "• **ಭತ್ತ**: 120 ಕೆಜಿ N/ಹೆಕ್ಟೇರ್ (3 ಕಂತುಗಳಲ್ಲಿ):\n"
                "  - 50% (60 ಕೆಜಿ) ನಾಟಿ ಮಾಡುವಾಗ\n"
                "  - 33% (40 ಕೆಜಿ) ತೆನೆ ಒಡೆಯುವಾಗ (21-25 ದಿನ)\n"
                "  - 17% (20 ಕೆಜಿ) ಹೂವು ಬಿಡುವಾಗ\n"
                "• **ಗೋಧಿ**: 120 ಕೆಜಿ N/ಹೆಕ್ಟೇರ್ (3 ಕಂತುಗಳಲ್ಲಿ)\n"
                "• **ಮೆಕ್ಕೆಜೋಳ**: 150 ಕೆಜಿ N/ಹೆಕ್ಟೇರ್ (3 ಸಮಾನ ಕಂತುಗಳಲ್ಲಿ)\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Telugu": (
                "**ICAR నత్రజని (Nitrogen) ఎరువుల సిఫార్సు:**\n\n"
                "• **వరి**: 120 కిలోల N/హెక్టారు (3 విడతలలో):\n"
                "  - 50% (60 కిలోలు) నాట్లు వేసే సమయంలో బేసల్ డోస్\n"
                "  - 33% (40 కిలోలు) పిలకల దశలో (20-25 రోజుల తర్వాత)\n"
                "  - 17% (20 కిలోలు) చిరుపొట్ట దశలో\n"
                "• **గోధుమ**: 120 కిలోల N/హెక్టారు\n"
                "• **మొక్కజొన్న**: 150 కిలోల N/హెక్టారు (3 సమాన విడతలలో)\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Tamil": (
                "**ICAR பரிந்துரைத்த நைட்ரஜன் (Nitrogen) உர அளவு:**\n\n"
                "• **நெல்**: 120 கிலோ N/ஹெக்டேர் (3 தவணைகளில்):\n"
                "  - 50% (60 கிலோ) நடவு செய்யும் போது அடி உரமாக\n"
                "  - 33% (40 கிலோ) தூர் கட்டும் பருவத்தில் (20-25 நாட்கள்)\n"
                "  - 17% (20 கிலோ) கதிர் உருவாகும் பருவத்தில்\n"
                "• **கோதுமை**: 120 கிலோ N/ஹெக்டேர்\n"
                "• **மக்காச்சோளம்**: 150 கிலோ N/ஹெக்டேர்\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Marathi": (
                "**ICAR शिफारस केलेले नत्र (युरिया) व्यवस्थापन:**\n\n"
                "• **भात**: 120 किलो N/हेक्टर (3 हप्त्यांमध्ये):\n"
                "  - 50% (60 किलो) लावणीच्या वेळी\n"
                "  - 33% (40 किलो) फुटवे फुटताना (20-25 दिवसांनी)\n"
                "  - 17% (20 किलो) लोंबी बाहेर पडताना\n"
                "• **गहू**: 120 किलो N/हेक्टर\n"
                "• **मका**: 150 किलो N/हेक्टर\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Bengali": (
                "**ICAR প্রস্তাবিত নাইট্রোজেন (ইউরিয়া) সার প্রয়োগ:**\n\n"
                "• **ধান**: ১২০ কেজি N/হেক্টর (৩ কিস্তিতে):\n"
                "  - ৫০% (৬০ কেজি) চারা রোপণের সময়\n"
                "  - ৩৩% (৪০ কেজি) কুশি গজানোর সময় (২০-২৫ দিন পর)\n"
                "  - ১৭% (২০ কেজি) থোর আসার সময়\n"
                "• **গম**: ১২০ কেজি N/হেক্টর\n"
                "• **ভুট্টা**: ১৫০ কেজি N/হেক্টর\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Gujarati": (
                "**ICAR ભલામણ કરેલ નાઇટ્રોજન (યુરિયા) વ્યવસ્થાપન:**\n\n"
                "• **ડાંગર**: 120 કિગ્રા N/હેક્ટર (3 હપ્તામાં)\n"
                "• **ઘઉં**: 120 કિગ્રા N/હેક્ટર\n"
                "• **મકાઈ**: 150 કિગ્રા N/હેક્ટર\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "Malayalam": (
                "**ICAR നൈട്രജൻ (യൂറിയ) വളപ്രയോഗം:**\n\n"
                "• **നെല്ല്**: 120 കിലോഗ്രാം N/ഹെക്ടർ (3 തവണകളായി):\n"
                "  - 50% നടീൽ സമയത്ത്\n"
                "  - 33% ചിനപ്പ് പൊട്ടുമ്പോൾ (20-25 ദിവസത്തിന് ശേഷം)\n"
                "  - 17% കതിര് വരുമ്പോൾ\n"
                "• **ഗോതമ്പ്**: 120 കിലോഗ്രാം N/ഹെക്ടർ\n"
                "• **ചോളം**: 150 കിലോഗ്രാം N/ഹെക്ടർ\n\n"
                "[Source: ICAR_Nitrogen_Management_Guidelines.pdf]"
            ),
            "source": "ICAR_Nitrogen_Management_Guidelines.pdf"
        },
        "ph": {
            "keywords": ["ph", "acidic", "alkaline", "saline", "soil health", "lime", "gypsum", "मिट्टी का ph", "ಮಣ್ಣಿನ ph", "నేల ph", "மண் ph", "सामू"],
            "English": (
                "**Optimal Soil pH Guidelines by Crop (ICAR):**\n\n"
                "• **Rice**: Optimal pH **5.5 – 7.0** (Tolerates slightly acidic to neutral soils)\n"
                "• **Wheat & Maize**: Optimal pH **6.0 – 7.5**\n"
                "• **Cotton**: Optimal pH **6.0 – 8.0**\n"
                "• **Tomato & Potato**: Optimal pH **5.5 – 6.5**\n\n"
                "🛠️ **Correction Measures**:\n"
                "• **Acidic Soil (pH < 6.0)**: Apply agricultural lime ($\text{CaCO}_3$) @ 2–4 tonnes/ha to increase pH.\n"
                "• **Alkaline/Sodic Soil (pH > 8.0)**: Apply gypsum ($\text{CaSO}_4\cdot2\text{H}_2\text{O}$) @ 5 tonnes/ha followed by flooding and green manuring.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Hindi": (
                "**फसलों के लिए उपयुक्त मिट्टी का pH (ICAR):**\n\n"
                "• **चावल (धान)**: pH **5.5 – 7.0**\n"
                "• **गेहूं और मक्का**: pH **6.0 – 7.5**\n"
                "• **कपास**: pH **6.0 – 8.0**\n"
                "• **टमाटर और आलू**: pH **5.5 – 6.5**\n\n"
                "🛠️ **सुधार के उपाय**:\n"
                "• **अम्लीय मिट्टी (pH < 6.0)**: 2-4 टन/हेक्टेयर चूना (Lime) डालें।\n"
                "• **क्षारीय मिट्टी (pH > 8.0)**: 5 टन/हेक्टेयर जिप्सम डालें और हरी खाद लगाएं।\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Kannada": (
                "**ಬೆಳೆಗಳಿಗೆ ಸೂಕ್ತವಾದ ಮಣ್ಣಿನ pH ಮೌಲ್ಯ (ICAR):**\n\n"
                "• **ಭತ್ತ**: pH **5.5 – 7.0**\n"
                "• **ಗೋಧಿ ಮತ್ತು ಮೆಕ್ಕೆಜೋಳ**: pH **6.0 – 7.5**\n"
                "• **ಹತ್ತಿ**: pH **6.0 – 8.0**\n"
                "• **ಟೊಮೇಟೊ**: pH **5.5 – 6.5**\n\n"
                "🛠️ **ಮಣ್ಣಿನ ಸುಧಾರಣೆ**:\n"
                "• **ಆಮ್ಲೀಯ ಮಣ್ಣು (pH < 6.0)**: ಎಕರೆಗೆ ಸುಣ್ಣ (Lime) ಹಾಕಿ.\n"
                "• **ಕ್ಷಾರೀಯ ಮಣ್ಣು (pH > 8.0)**: ಹೆಕ್ಟೇರ್‌ಗೆ 5 ಟನ್ ಜಿಪ್ಸಮ್ (Gypsum) ಬಳಸಿ.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Telugu": (
                "**పంటలకు సరైన నేల pH మార్గదర్శకాలు (ICAR):**\n\n"
                "• **వరి**: pH **5.5 – 7.0**\n"
                "• **గోధుమ & మొక్కజొన్న**: pH **6.0 – 7.5**\n"
                "• **పత్తి**: pH **6.0 – 8.0**\n"
                "• **టమాటా & బంగాళాదుంప**: pH **5.5 – 6.5**\n\n"
                "🛠️ **దిద్దుబాటు చర్యలు**:\n"
                "• **ఆమ్ల నేలలు (pH < 6.0)**: హెక్టారుకు 2-4 టన్నుల సున్నం వేయండి.\n"
                "• **క్షార నేలలు (pH > 8.0)**: హెక్టారుకు 5 టన్నుల జిప్సం వేసి పచ్చిరొట్ట ఎరువులు వాడండి.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Tamil": (
                "**பயிர்களுக்கான உகந்த மண் pH அளவு (ICAR):**\n\n"
                "• **நெல்**: pH **5.5 – 7.0**\n"
                "• **கோதுமை & மக்காச்சோளம்**: pH **6.0 – 7.5**\n"
                "• **பருத்தி**: pH **6.0 – 8.0**\n"
                "• **தக்காளி & உருளைக்கிழங்கு**: pH **5.5 – 6.5**\n\n"
                "🛠️ **மண் திருத்தம்**:\n"
                "• **அமில மண் (pH < 6.0)**: எக்டேருக்கு 2-4 டன் விவசாய சுண்ணாம்பு இடவும்.\n"
                "• **கார மண் (pH > 8.0)**: எக்டேருக்கு 5 டன் ஜிப்சம் இடவும்.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Marathi": (
                "**पिकांसाठी जमिनीचा योग्य सामू (pH) (ICAR):**\n\n"
                "• **भात**: pH **5.5 – 7.0**\n"
                "• **गहू आणि मका**: pH **6.0 – 7.5**\n"
                "• **कापूस**: pH **6.0 – 8.0**\n\n"
                "🛠️ **सुधारणा उपाय**:\n"
                "• **आम्लधर्मी जमीन (pH < 6.0)**: २-४ टन/हेक्टर चुना वापरा.\n"
                "• **क्षारयुक्त जमीन (pH > 8.0)**: ५ टन/हेक्टर जिप्सम वापरा आणि ताग/धैंचाचे हिरवळीचे खत द्या.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Bengali": (
                "**ফসলের জন্য মাটির উপযুক্ত pH মান (ICAR):**\n\n"
                "• **ধান**: pH **5.5 – 7.0**\n"
                "• **গম ও ভুট্টা**: pH **6.0 – 7.5**\n"
                "• **আলু ও টমেটো**: pH **5.5 – 6.5**\n\n"
                "🛠️ **মাটি শোধনের উপায়**:\n"
                "• **অম্লীয় মাটি (pH < 6.0)**: হেক্টর প্রতি ২-৪ টন চুন প্রয়োগ করুন।\n"
                "• **ক্ষারীয় মাটি (pH > 8.0)**: হেক্টর প্রতি ৫ টন জিপসাম ও ধৈঞ্চা সবুজ সার ব্যবহার করুন।\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Gujarati": (
                "**પાક માટે જમીનનો યોગ્ય pH (ICAR):**\n\n"
                "• **ડાંગર**: pH **5.5 – 7.0**\n"
                "• **ઘઉં અને મકાઈ**: pH **6.0 – 7.5**\n"
                "• **કપાસ**: pH **6.0 – 8.0**\n\n"
                "🛠️ **સુધારણા પગલાં**:\n"
                "• **એસિડિક જમીન (pH < 6.0)**: ચૂનો ઉમેરો.\n"
                "• **ક્ષારીય જમીન (pH > 8.0)**: હેક્ટર દીઠ 5 ટન જીપ્સમ ઉમેરો.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "Malayalam": (
                "**വിളകൾക്ക് അനുയോജ്യമായ മണ്ണിന്റെ pH (ICAR):**\n\n"
                "• **നെല്ല്**: pH **5.5 – 7.0**\n"
                "• **പച്ചക്കറികൾ**: pH **6.0 – 7.0**\n\n"
                "🛠️ **മണ്ണ് പരിപാലനം**:\n"
                "• **അമ്ല മണ്ണ് (pH < 6.0)**: കുമ്മായം ചേർക്കുക.\n"
                "• **ക്ഷാര മണ്ണ് (pH > 8.0)**: ജിപ്സം ഉപയോഗിക്കുക.\n\n"
                "[Source: ICAR_Soil_Health_pH_Management.pdf]"
            ),
            "source": "ICAR_Soil_Health_pH_Management.pdf"
        },
        "irrigation": {
            "keywords": ["irrigate", "irrigation", "moisture", "water", "when to water", "et0", "सिंचाई", "पानी", "ನೀರಾವರಿ", "ನೀರು", "నీరు", "నీటిపారుదల", "பாசனம்", "पाणी"],
            "English": (
                "**Critical Irrigation Stages & Moisture Thresholds (ICAR & FAO-56):**\n\n"
                "• **Rice**: Maintain 2–5 cm standing water during transplanting, tillering, panicle initiation, and flowering. Soil moisture threshold: **> 80%**.\n"
                "• **Wheat**: 6 critical stages — Crown Root Initiation (CRI at 20-25 DAS, most critical!), tillering, jointing, flowering, milk stage, and dough stage. Moisture threshold: **> 50%**.\n"
                "• **Maize**: Most sensitive during knee-high, tasseling, and silking stages. Moisture threshold: **> 55%**.\n"
                "• **Cotton**: Critical at flowering and boll development. Moisture threshold: **> 45%**.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf & FAO_Crop_Evapotranspiration_Paper56.pdf]"
            ),
            "Hindi": (
                "**महत्वपूर्ण सिंचाई अवस्थाएं (ICAR और FAO-56):**\n\n"
                "• **चावल**: कल्ले फूटने और बाली आने के समय 2-5 सेमी पानी बनाए रखें (नमी **> 80%**)।\n"
                "• **गेहूं**: सबसे महत्वपूर्ण सिंचाई **CRI अवस्था (बुवाई के 20-25 दिन बाद)** पर करें। इसके बाद कल्ले फूटते समय और फूल आने पर सिंचाई करें।\n"
                "• **मक्का**: घुटने की ऊंचाई और भुट्टा बनते समय सिंचाई आवश्यक है।\n"
                "• **कपास**: फूल और टिंडे बनते समय पानी की कमी न होने दें।\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Kannada": (
                "**ಪ್ರಮುಖ ನೀರಾವರಿ ಹಂತಗಳು (ICAR & FAO-56):**\n\n"
                "• **ಭತ್ತ**: ನಾಟಿ, ತೆನೆ ಒಡೆಯುವ ಮತ್ತು ಹೂವು ಬಿಡುವ ಹಂತಗಳಲ್ಲಿ 2-5 ಸೆಂ.ಮೀ ನೀರು ನಿಲ್ಲಿಸಿ (ತೇವಾಂಶ **> 80%**).\n"
                "• **ಗೋಧಿ**: ಬಿತ್ತನೆಯ 20-25 ದಿನಗಳಲ್ಲಿ (CRI ಹಂತ) ಮೊದಲ ಮತ್ತು ಅತಿ ಮುಖ್ಯ ನೀರಾವರಿ ನೀಡಿ.\n"
                "• **ಮೆಕ್ಕೆಜೋಳ**: ತೆನೆ ಬರುವ ಮತ್ತು ಕಾಳು ಕಟ್ಟುವ ಹಂತದಲ್ಲಿ ನೀರು ನೀಡಿ.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Telugu": (
                "**ముఖ్యమైన నీటిపారుదల దశలు (ICAR & FAO-56):**\n\n"
                "• **వరి**: పిలకల దశ, చిరుపొట్ట మరియు పూత దశలలో 2-5 సెం.మీ నీరు నిలపాలి (తేమ **> 80%**).\n"
                "• **గోధుమ**: విత్తిన 20-25 రోజులకు (CRI దశ) మొదటి మరియు ముఖ్యమైన తడి ఇవ్వాలి.\n"
                "• **మొక్కజొన్న**: మోకాళ్ళ ఎత్తు మరియు పూత దశలో నీరు అందించాలి.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Tamil": (
                "**முக்கியமான பாசன நிலைகள் (ICAR & FAO-56):**\n\n"
                "• **நெல்**: தூர் கட்டும் பருவம் மற்றும் கதிர் வரும் பருவத்தில் 2-5 செ.மீ நீர் தேக்கி வைக்கவும் (ஈரப்பதம் **> 80%**).\n"
                "• **கோதுமை**: விதைத்த 20-25 நாட்களில் (CRI நிலை) முதல் பாசனம் செய்ய வேண்டும்.\n"
                "• **மக்காச்சோளம்**: பூக்கும் பருவம் மற்றும் கதிர் உருவாகும் போது பாசனம் அவசியம்.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Marathi": (
                "**पाणी व्यवस्थापनाच्या महत्त्वाच्या अवस्था (ICAR & FAO-56):**\n\n"
                "• **भात**: फुटवे फुटताना आणि लोंबी भरताना २-५ सेमी पाणी ठेवावे (ओलावा **> ८०%**).\n"
                "• **गहू**: मुकुट मुळे फुटण्याच्या अवस्थेत (CRI - २०-२५ दिवसांनी) पहिले पाणी द्यावे.\n"
                "• **मका**: गुडघाभर उंची आणि तुरा येताना पाणी द्यावे.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Bengali": (
                "**গুরুত্বপূর্ণ সেচ পর্যায় (ICAR ও FAO-56):**\n\n"
                "• **ধান**: চারা রোপণ ও কুশি গজানোর সময় ২-৫ সেমি পানি ধরে রাখুন (আর্দ্রতা **> ৮০%**)।\n"
                "• **গম**: বীজ বপনের ২০-২৫ দিন পর (CRI পর্যায়) প্রথম ও প্রধান সেচ দিন।\n"
                "• **ভুট্টা**: হাঁটু সমান উচ্চতা এবং মোচা আসার সময় সেচ দেওয়া প্রয়োজন।\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Gujarati": (
                "**મહત્વના પિયત તબક્કા (ICAR & FAO-56):**\n\n"
                "• **ડાંગર**: કંઠી ફૂટતી વખતે 2-5 સેમી પાણી ભરેલું રાખો.\n"
                "• **ઘઉં**: વાવણી પછીના 20-25 દિવસે (CRI સ્ટેજ) પ્રથમ પિયત આપો.\n"
                "• **મકાઈ**: ઘૂંટણ જેટલી ઊંચાઈ અને ડોડા બેસતી વખતે પાણી આપવું.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "Malayalam": (
                "**നിർണ്ണായക ജലസേചന ഘട്ടങ്ങൾ (ICAR & FAO-56):**\n\n"
                "• **നെല്ല്**: ചിനപ്പ് പൊട്ടുമ്പോഴും കതിര് വരുമ്പോഴും 2-5 സെ.മീ വെള്ളം നിർത്തുക.\n"
                "• **ഗോതമ്പ്**: വിതച്ച് 20-25 ദിവസത്തിനകം (CRI ഘട്ടം) നനയ്ക്കുക.\n\n"
                "[Source: ICAR_Irrigation_Water_Management.pdf]"
            ),
            "source": "ICAR_Irrigation_Water_Management.pdf"
        },
        "yield": {
            "keywords": ["yield", "production", "target", "forecast", "how much yield", "उपज", "उत्पादन", "ಇಳುವರಿ", "దిగుబడి", "மகசூல்", "उत्पन्न", "ফলন"],
            "English": (
                "**Achievable Target Yields in India (ICAR Benchmarks):**\n\n"
                "• **Rice**: National Average **2.6 t/ha** | Achievable Target **5.5 – 6.5 t/ha**\n"
                "• **Wheat**: National Average **3.5 t/ha** | Achievable Target **5.0 – 6.0 t/ha**\n"
                "• **Maize**: National Average **3.0 t/ha** | Achievable Target **7.0 – 8.5 t/ha**\n"
                "• **Cotton**: National Average **0.48 t/ha** lint | Achievable Target **0.8 – 1.2 t/ha** lint\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Hindi": (
                "**भारत में प्रमुख फसलों का संभावित उत्पादन लक्ष्य (ICAR):**\n\n"
                "• **चावल**: राष्ट्रीय औसत 2.6 टन/हेक्टेयर | **लक्ष्य: 5.5 – 6.5 टन/हेक्टेयर**\n"
                "• **गेहूं**: राष्ट्रीय औसत 3.5 टन/हेक्टेयर | **लक्ष्य: 5.0 – 6.0 टन/हेक्टेयर**\n"
                "• **मक्का**: राष्ट्रीय औसत 3.0 टन/हेक्टेयर | **लक्ष्य: 7.0 – 8.5 टन/हेक्टेयर**\n"
                "• **कपास**: **0.8 – 1.2 टन/हेक्टेयर**\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Kannada": (
                "**ಪ್ರಮುಖ ಬೆಳೆಗಳ ಗರಿಷ್ಠ ಇಳುವರಿ ಗುರಿ (ICAR):**\n\n"
                "• **ಭತ್ತ**: ರಾಷ್ಟ್ರೀಯ ಸರಾಸರಿ 2.6 ಟನ್ | **ಗುರಿ: 5.5 – 6.5 ಟನ್/ಹೆಕ್ಟೇರ್**\n"
                "• **ಗೋಧಿ**: **5.0 – 6.0 ಟನ್/ಹೆಕ್ಟೇರ್**\n"
                "• **ಮೆಕ್ಕೆಜೋಳ**: **7.0 – 8.5 ಟನ್/ಹೆಕ್ಟೇರ್**\n"
                "• **ಕಬ್ಬು**: **100 – 120 ಟನ್/ಹೆಕ್ಟೇರ್**\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Telugu": (
                "**భారతదేశంలో ప్రధాన పంటల దిగుబడి లక్ష్యాలు (ICAR):**\n\n"
                "• **వరి**: జాతీయ సగటు 2.6 టన్నులు | **లక్ష్యం: 5.5 – 6.5 టన్నులు/హెక్టారు**\n"
                "• **గోధుమ**: **5.0 – 6.0 టన్నులు/హెక్టారు**\n"
                "• **మొక్కజొన్న**: **7.0 – 8.5 టన్నులు/హెక్టారు**\n"
                "• **పత్తి**: **0.8 – 1.2 టన్నులు/హెక్టారు**\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Tamil": (
                "**இந்தியாவில் முக்கிய பயிர்களின் இலக்கு மகசூல் (ICAR):**\n\n"
                "• **நெல்**: தேசிய சராசரி 2.6 டன் | **இலக்கு: 5.5 – 6.5 டன்/ஹெக்டேர்**\n"
                "• **கோதுமை**: **5.0 – 6.0 டன்/ஹெக்டேர்**\n"
                "• **மக்காச்சோளம்**: **7.0 – 8.5 டன்/ஹெக்டேர்**\n"
                "• **பருத்தி**: **0.8 – 1.2 டன்/ஹெக்டேர்**\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Marathi": (
                "**भारतातील मुख्य पिकांचे संभाव्य उत्पादन उद्दिष्ट (ICAR):**\n\n"
                "• **भात**: राष्ट्रीय सरासरी २.६ टन | **उद्दिष्ट: ५.५ – ६.५ टन/हेक्टर**\n"
                "• **गहू**: **५.० – ६.० टन/हेक्टर**\n"
                "• **मका**: **७.० – ८.५ टन/हेक्टर**\n"
                "• **कापूस**: **०.८ – १.२ टन/हेक्टर**\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Bengali": (
                "**ভারতে প্রধান ফসলের সম্ভাব্য ফলন লক্ষ্যমাত্রা (ICAR):**\n\n"
                "• **ধান**: জাতীয় গড় ২.৬ টন | **লক্ষ্যমাত্রা: ৫.৫ – ৬.৫ টন/হেক্টর**\n"
                "• **গম**: **৫.০ – ৬.০ টন/হেক্টর**\n"
                "• **ভুট্টা**: **৭.০ – ৮.৫ টন/হেক্টর**\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Gujarati": (
                "**ભારતમાં મુખ્ય પાકોનું ઉત્પાદન લક્ષ્ય (ICAR):**\n\n"
                "• **ડાંગર**: લક્ષ્ય: 5.5 – 6.5 ટન/હેક્ટર\n"
                "• **ઘઉં**: 5.0 – 6.0 ટન/હેક્ટર\n"
                "• **મકાઈ**: 7.0 – 8.5 ટન/હેક્ટર\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Malayalam": (
                "**പ്രധാന വിളകളുടെ ഉൽപ്പാദന ലക്ഷ്യങ്ങൾ (ICAR):**\n\n"
                "• **നെല്ല്**: ലക്ഷ്യം: 5.5 – 6.5 ടൺ/ഹെക്ടർ\n"
                "• **ഗോതമ്പ്**: 5.0 – 6.0 ടൺ/ഹെക്ടർ\n"
                "• **ചോളം**: 7.0 – 8.5 ടൺ/ഹെക്ടർ\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "source": "ICAR_Yield_Benchmarks_Crop_Statistics.pdf"
        },
        "crops": {
            "keywords": ["crop", "crops", "recommend", "which crop", "best crop", "suitability", "फसल", "ಬೆಳೆ", "పంట", "பயிர்", "पीक", "ফসল"],
            "English": (
                "**Optimal Crop Selection Guide by Soil & Climate (ICAR):**\n\n"
                "• **Black Cotton Soil (Vertisol)**: Cotton, Soybean, Chickpea, Pigeonpea, Sorghum\n"
                "• **Red Loamy Soil (Alfisol)**: Groundnut, Maize, Ragi (Finger Millet), Pulses, Tomato\n"
                "• **Alluvial Soil**: Rice, Wheat, Sugarcane, Mustard, Potato\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Hindi": (
                "**मिट्टी के प्रकार के अनुसार उपयुक्त फसलें:**\n\n"
                "• **काली मिट्टी**: कपास, सोयाबीन, चना, अरहर, ज्वार\n"
                "• **लाल मिट्टी**: मूंगफली, मक्का, रागी, दालें, टमाटर\n"
                "• **जलोढ़ (दोमट) मिट्टी**: चावल, गेहूं, गन्ना, सरसों, आलू\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Kannada": (
                "**ಮಣ್ಣಿನ ಪ್ರಕಾರಕ್ಕೆ ಅನುಗುಣವಾಗಿ ಸೂಕ್ತ ಬೆಳೆಗಳು:**\n\n"
                "• **ಕಪ್ಪು ಮಣ್ಣು**: ಹತ್ತಿ, ಸೋಯಾಬೀನ್, ಕಡಲೆ, ತೊಗರಿ, ಜೋಳ\n"
                "• **ಕೆಂಪು ಮಣ್ಣು**: ಕಡಲೆಕಾಯಿ, ಮೆಕ್ಕೆಜೋಳ, ರಾಗಿ, ಟೊಮೇಟೊ\n"
                "• **ಮೆಕ್ಕಲು ಮಣ್ಣು**: ಭತ್ತ, ಗೋಧಿ, ಕಬ್ಬು, ಸಾಸಿವೆ\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Telugu": (
                "**నేల రకాన్ని బట్టి అనువైన పంటలు:**\n\n"
                "• **నల్లరేగడి నేలలు**: పత్తి, సోయాబీన్, శనగలు, కందులు, జొన్నలు\n"
                "• **ఎర్ర నేలలు**: వేరుశనగ, మొక్కజొన్న, రాగులు, పప్పుధాన్యాలు, టమాటా\n"
                "• **ఒండ్రు నేలలు**: వరి, గోధుమ, చెరకు, ఆవాలు\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Tamil": (
                "**மண் வகைகளுக்கு ஏற்ப ஏற்ற பயிர்கள்:**\n\n"
                "• **கரிசல் மண்**: பருத்தி, சோயாபீன்ஸ், கொண்டைக்கடலை, துவரை, சோளம்\n"
                "• **செம்மண்**: நிலக்கடலை, மக்காச்சோளம், கேழ்வரகு, பருப்பு வகைகள், தக்காளி\n"
                "• **வண்டல் மண்**: நெல், கோதுமை, கரும்பு, கடுகு\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Marathi": (
                "**जमिनीच्या प्रकारानुसार योग्य पिके:**\n\n"
                "• **काळी जमीन (रेगूर)**: कापूस, सोयाबीन, हरभरा, तूर, ज्वारी\n"
                "• **तांबडी जमीन**: भुईमूग, मका, नाचणी, डाळी, टोमॅटो\n"
                "• **गाळाची जमीन**: भात, गहू, ऊस, मोहरी, बटाटा\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Bengali": (
                "**মাটির ধরন অনুযায়ী উপযোগী ফসল:**\n\n"
                "• **কালো মাটি**: তুলা, সয়াবিন, ছোলা, অড়হর, জোয়ার\n"
                "• **লাল মাটি**: চীনাবাদাম, ভুট্টা, রাগি, ডাল, টমেটো\n"
                "• **পলি মাটি**: ধান, গম, আখ, সরিষা, আলু\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Gujarati": (
                "**જમીનના પ્રકાર મુજબ યોગ્ય પાક:**\n\n"
                "• **કાળી જમીન**: કપાસ, સોયાબીન, ચણા, તુવેર, જુવાર\n"
                "• **લાલ જમીન**: મગફળી, મકાઈ, રાગી, કઠોળ, ટામેટા\n"
                "• **ગોરાડુ/કાંપવાળી જમીન**: ડાંગર, ઘઉં, શેરડી, રાયડો\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "Malayalam": (
                "**മണ്ണിന്റെ തരത്തിന് അനുയോജ്യമായ വിളകൾ:**\n\n"
                "• **കറുത്ത മണ്ണ്**: പരുത്തി, സോയാബീൻ, കടല, ചോളം\n"
                "• **ചുവന്ന മണ്ണ്**: നിലക്കടല, ചോളം, റാഗി, തക്കാളി\n"
                "• **എക്കൽ മണ്ണ്**: നെല്ല്, ഗോതമ്പ്, കരിമ്പ്\n\n"
                "[Source: ICAR_Yield_Benchmarks_Crop_Statistics.pdf]"
            ),
            "source": "ICAR_Yield_Benchmarks_Crop_Statistics.pdf"
        }
    }

    @classmethod
    def match(cls, query: str, language: str) -> dict | None:
        q_clean = query.lower().strip()
        for cat, data in cls.KNOWLEDGE.items():
            for kw in data["keywords"]:
                if kw in q_clean:
                    ans = data.get(language, data["English"])
                    return {
                        "answer": ans,
                        "citations": [data["source"]],
                        "confidence": "High (ICAR Verified Knowledge)",
                        "language": language,
                    }
        return None


class RAGService:
    _vectorstore = None
    _chain = None

    @classmethod
    def _build_vectorstore(cls):
        """Build ChromaDB vectorstore if GOOGLE_API_KEY is available."""
        if not settings.GOOGLE_API_KEY or GoogleGenerativeAIEmbeddings is None or Chroma is None:
            return

        persist_dir = settings.CHROMA_PERSIST_DIR
        docs_dir = Path(settings.RAG_DOCS_DIR)
        os.makedirs(persist_dir, exist_ok=True)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GOOGLE_API_KEY,
        )

        if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            cls._vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
            )
            return

        all_docs = []
        if docs_dir.exists() and PyMuPDFLoader is not None:
            for pdf_file in docs_dir.glob("*.pdf"):
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                    pages = loader.load()
                    for page in pages:
                        page.metadata["source"] = pdf_file.name
                    all_docs.extend(pages)
                except Exception:
                    pass

        if all_docs and RecursiveCharacterTextSplitter is not None:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = splitter.split_documents(all_docs)
            cls._vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
            cls._vectorstore.persist()

    @classmethod
    def _get_chain(cls, language: str = "English"):
        if not settings.GOOGLE_API_KEY:
            return None

        if cls._vectorstore is None:
            cls._build_vectorstore()

        if cls._vectorstore is None or ChatGoogleGenerativeAI is None or ChatPromptTemplate is None:
            return None

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )
        retriever = cls._vectorstore.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_template(
            SYSTEM_PROMPT.replace("{language}", language)
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str, language: str = "English", field_context: dict | None = None) -> dict:
        """
        Main query entrypoint:
        1. Checks for casual greetings / conversational queries -> Friendly introduction
        2. Tries Gemini online RAG if GOOGLE_API_KEY is configured
        3. Matches curated ICAR/FAO knowledge base with clean formatting & exact citations
        4. If outside knowledge base, provides a polite fallback
        """
        q_raw = question.strip().lower()
        q_first_word = q_raw.split()[0] if q_raw.split() else q_raw

        # 1. Intent: Greeting
        if (
            q_raw in GREETING_KEYWORDS
            or q_first_word in GREETING_KEYWORDS
            or any(w in q_raw for w in ["good morning", "good afternoon", "who are you", "what can you do", "help", "who is agri"])
        ):
            return {
                "answer": GREETING_RESPONSES.get(language, GREETING_RESPONSES["English"]),
                "citations": [],
                "confidence": "High",
                "language": language,
            }

        # Intent: Gratitude
        if (
            q_raw in THANKS_KEYWORDS
            or any(t in q_raw for t in ["thank you", "thanks", "dhanyawad", "shukriya", "ಧನ್ಯವಾದಗಳು"])
        ):
            return {
                "answer": THANKS_RESPONSES.get(language, THANKS_RESPONSES["English"]),
                "citations": [],
                "confidence": "High",
                "language": language,
            }

        # 2. Try Gemini Online Chain if API key is set
        enriched_question = question
        if field_context:
            ctx = (
                f"[Field Context: Moisture={field_context.get('moisture')}%, "
                f"pH={field_context.get('ph')}, N={field_context.get('nitrogen')} mg/kg, "
                f"Crop={field_context.get('crop_type')}, Stage={field_context.get('growth_stage')}]"
            )
            enriched_question = f"{ctx}\n{question}"

        chain = self._get_chain(language)
        if chain is not None:
            try:
                answer = chain.invoke(enriched_question)
                return {
                    "answer": answer,
                    "citations": [],
                    "confidence": "High (Gemini Online RAG)",
                    "language": language,
                }
            except Exception as e:
                print(f"Gemini API query failed ({e}), falling back to curated offline knowledge base...")

        # 3. Curated ICAR & FAO Knowledge Match
        matched = KnowledgeBase.match(question, language)
        if matched:
            return matched

        # 4. Out-of-knowledge fallback
        fallbacks = {
            "English": "I don't have verified data on that specific query in the ICAR/FAO knowledge base. Please consult your local Krishi Vigyan Kendra (KVK) or extension officer for field-specific advice. [Source: ICAR Package of Practices]",
            "Hindi": "इस विशिष्ट प्रश्न पर ICAR/FAO संदर्भ में जानकारी उपलब्ध नहीं है। कृपया अधिक जानकारी के लिए अपने स्थानीय कृषि विज्ञान केंद्र (KVK) से संपर्क करें। [Source: ICAR Package of Practices]",
            "Kannada": "ಈ ನಿರ್ದಿಷ್ಟ ಪ್ರಶ್ನೆಗೆ ICAR/FAO ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರವನ್ನು (KVK) ಸಂಪರ್ಕಿಸಿ. [Source: ICAR Package of Practices]",
            "Telugu": "ఈ నిర్దిష్ట ప్రశ్నకు ICAR/FAO లో సమాచారం అందుబాటులో లేదు. దయచేసి మరిన్ని వివరాల కోసం మీ సమీప కృషి విజ్ఞాన కేంద్రాన్ని (KVK) సంప్రదించండి. [Source: ICAR Package of Practices]",
            "Tamil": "இந்த குறிப்பிட்ட கேள்விக்கு ICAR/FAO தகவல் தளத்தில் தகவல் இல்லை. கூடுதல் தகவலுக்கு உங்கள் அருகிலுள்ள வேளாண் அறிவியல் மையத்தை (KVK) அணுகவும். [Source: ICAR Package of Practices]",
            "Marathi": "या विशिष्ट प्रश्नावर ICAR/FAO संदर्भामध्ये माहिती उपलब्ध नाही. कृपया अधिक माहितीसाठी तुमच्या जवळच्या कृषी विज्ञान केंद्राशी (KVK) संपर्क साधा. [Source: ICAR Package of Practices]",
            "Bengali": "এই নির্দিষ্ট প্রশ্নে ICAR/FAO ডাটাবেসে তথ্য নেই। বিস্তারিত তথ্যের জন্য আপনার নিকটস্থ কৃষি বিজ্ঞান কেন্দ্রের (KVK) সাথে যোগাযোগ করুন। [Source: ICAR Package of Practices]",
            "Gujarati": "આ પ્રશ્ન અંગે ICAR/FAO માર્ગદર્શિકામાં માહિતી ઉપલબ્ધ નથી. વધુ માહિતી માટે કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) નો સંપર્ક કરો. [Source: ICAR Package of Practices]",
            "Malayalam": "ഈ പ്രത്യേക ചോദ്യത്തിന് ICAR/FAO ഡാറ്റാബേസിൽ വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി നിങ്ങളുടെ അടുത്തുള്ള കൃഷി വിജ്ഞാന കേന്ദ്രവുമായി (KVK) ബന്ധപ്പെടുക. [Source: ICAR Package of Practices]",
        }
        return {
            "answer": fallbacks.get(language, fallbacks["English"]),
            "citations": ["ICAR Package of Practices"],
            "confidence": "Low",
            "language": language,
        }


rag_service = RAGService()
