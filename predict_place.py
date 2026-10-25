import sys
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from openai import OpenAI
import requests
import json

# Load model once at module level to avoid reloading on every request (if possible)
MODEL_PATH = "raigad_places_model.keras"
cnn_model = None

# Configure OpenRouter API from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

# In-Memory Cache for Instant Responses
PLACE_CACHE = {}
GENERAL_CHAT_CACHE = {}

def load_raigad_model():
    global cnn_model
    if cnn_model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            cnn_model = load_model(MODEL_PATH)
            try:
                # Model Warmup to eliminate first-request lag
                dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                cnn_model.predict(dummy_input)
                print("Raigad model warmed up successfully.")
            except Exception as we:
                print("Warmup notice:", we)
        else:
            print(f"Error: Model file '{MODEL_PATH}' not found.")
    return cnn_model

# Warm up model on import
try:
    load_raigad_model()
except Exception:
    pass

def get_class_names():
    class_names = []
    if os.path.exists("class_names.txt"):
        with open("class_names.txt", "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]

    else:
        # Fallback to directory structure if file doesn't exist
        train_dir = os.path.join("dataset", "Train")
        if os.path.exists(train_dir):
             class_names = sorted(os.listdir(train_dir))
    return class_names

# Rich Immersive Storytelling Knowledge Base for Raigad District (Zero-Downtime Offline Fallback)
RAIGAD_KNOWLEDGE_BASE = {
    "Raigad Fort Hirkani Buruj": {
        "English": "Perched on the sheer eastern precipice of Raigad Fort, where the basalt cliffs plunge thousands of feet into the misty Sahyadri valleys below, stands the legendary Hirkani Buruj. This bastion is celebrated not merely for its military might, but as an eternal monument to a mother's fearless devotion.\n\nIn the late 17th century, when Raigad was the glorious capital of Hindavi Swarajya, a humble milkmaid named Hirkani climbed the fort daily from the foothills of Valivande to sell fresh milk. One fateful evening, she lingered too long near the royal quarters and reached the Maha Darwaja just as the colossal gates were bolted shut at sunset in accordance with strict military protocols. Desperate to return to her infant child crying in the dark village below, she begged the guards, but the royal laws of Swarajya permitted no exceptions. Driven by pure maternal love, Hirkani made a daring choice: under the shroud of darkness, she scaled down the sheer, razor-sharp cliff that was considered impossible for even seasoned mountain warriors.\n\nThe next morning, when she walked through the gates again with fresh milk, news of her astonishing feat reached Chhatrapati Shivaji Maharaj. The King summoned her to the royal court, listened in awe to her harrowing descent, and honored her with a royal shawl and gold bangles. Recognizing the vulnerability of that cliff, Shivaji Maharaj ordered a mighty defensive bastion to be erected on that exact precipice and named it 'Hirkani Buruj' in her eternal honor.",
        "Hindi": "सह्याद्रि पर्वत श्रृंखला के विशाल काले पाषाणों पर, जहां से नीचे हजारों फीट गहरी धुंधली घाटियां नजर आती हैं, वहां गर्व से खड़ा है ऐतिहासिक 'हिरकणी बुरुज'। यह बुरुज केवल युद्ध का मोर्चा नहीं, बल्कि एक मां के अदम्य साहस और निःस्वार्थ मातृप्रेम का अमर स्मारक है।\n\n17वीं सदी के स्वर्णिम काल में, जब रायगड हिंदवी स्वराज्य की राजधानी था, वळिवंडे गांव की 'हिरकणी' नाम की एक साधारण गवालन रोज दूध बेचने किले पर आती थी। एक दिन शाम को राजदरबार के कार्य में देर हो जाने के कारण, वह सूर्यास्त के बाद महादरवाजे पर पहुंची, लेकिन सैन्य नियमों के अनुसार द्वार बंद हो चुके थे। नीचे गांव में अपने भूखे और रोते हुए नन्हे बच्चे की याद में तड़पती हिरकणी ने प्रहरियों से बहुत विनती की, पर स्वराज्य के कड़े नियमों में कोई ढील नहीं दी जा सकती थी। तब अपने बच्चे के पास पहुंचने के दृढ़ संकल्प के साथ, उसने अंधेरी रात में उस सीधी और जानलेवा चट्टान से नीचे उतरने का असंभव निर्णय लिया, जिस पर चढ़ना या उतरना किसी कुशल योद्धा के लिए भी नामुमकिन माना जाता था।\n\nअगली सुबह जब वह पुनः दूध लेकर किले के मुख्य द्वार पर पहुंची, तो उसकी इस हैरतअंगेज यात्रा की खबर छत्रपति शिवाजी महाराज तक पहुंची। महाराज ने उसे राजदरबार में बुलाकर उसके अदम्य साहस को नमन किया, उसे स्वर्ण कंगन और जरी की साड़ी देकर सम्मानित किया। उस कड़े की सुरक्षा सुनिश्चित करने के लिए महाराज ने वहां एक भव्य और अभेद्य बुरुज का निर्माण कराया और उसे सदा के लिए 'हिरकणी बुरुज' नाम दिया।",
        "Marathi": "सह्याद्रीच्या उत्तुंग काळ्या पाषाणावर, जिथून खाली अथांग दर्‍या आणि धुक्याची चादर दिसते, तिथे डौलाने उभा आहे जगप्रसिद्ध 'हिरकणी बुरुज'. हा बुरुज केवळ एक लष्करी तटबंदी नसून, एका आईच्या अथांग आणि अलौकिक मातृप्रेमाचे अजरामर प्रतीक आहे.\n\n१७ व्या शतकात, जेव्हा दुर्गराज रायगड हिंदवी स्वराज्याची राजधानी म्हणून झळाळत होता, तेव्हा गडाच्या पायथ्याशी असलेल्या वाळिवंदे गावची 'हिरकणी' नावाची एक साधी गवळण रोज गडावर दूध विकायला येत असे. एका सायंकाळी राजवाड्यात दूध देता देता तिला उशीर झाला आणि सूर्यास्ताची तोफ डागली जाताच गडाचे महादरवाजे लष्करी नियमांनुसार कडेकोट बंद झाले. गडाखाली पाळण्यात रडणाऱ्या आपल्या तान्ह्या बाळाच्या काळजीने व्याकूळ झालेल्या हिरकणीने पहारेकऱ्यांना काकुळतीने दार उघडण्याची विनंती केली, परंतु छत्रपती शिवाजी महाराजांचे स्वराज्याचे नियम अत्यंत कडक होते. आपल्या बाळाच्या मायेपोटी तिने काळोख्या रात्री त्या अथांग, सरळ आणि अंगावर काटा आणणाऱ्या कड्यावरून खाली उतरण्याचा अशक्यप्राय निश्चय केला, जिथून उतरायला कसलेला मावळाही धजावत नसे.\n\nदुसऱ्या दिवशी सकाळी जेव्हा हिरकणी पुन्हा नेहमीप्रमाणे गडावर दूध घेऊन पोहोचली, तेव्हा पहारेकरी चकित झाले. ही वार्ता तात्काळ छत्रपती शिवाजी महाराजांपर्यंत गेली. शिवरायांनी तिला भर दरबारात बोलावून तिच्या पाठीवर कौतुकाची थाप ठेवली, साडी-चोळी आणि सोन्याचे कडे देऊन तिचा यथोचित गौरव केला. गडाच्या या बाजूने कोणीही शत्रू चढू नये म्हणून महाराजांनी त्या कड्यावर एक भक्कम बुरुज उभारला आणि त्या बुरुजाला नाव दिले 'हिरकणी बुरुज'!"
    },
    "Raigad Fort Jagadishwar Temple": {
        "English": "Nestled serenely on the royal plateau of Raigad Fort, the Jagadishwar Temple stands as an architectural and spiritual marvel dedicated to Lord Shiva. Crafted from polished black basalt, its grand structure reflects a timeless harmony of Maratha and Indo-Islamic dome architecture, symbolizing the inclusive vision of Hindavi Swarajya.\n\nChhatrapati Shivaji Maharaj began his mornings on the fort with prayers and meditation inside this sanctum, seeking divine blessings before steering the affairs of the Maratha Empire. The temple features a tranquil central sanctum, a magnificent Nandi pavilion, and intricately carved pillars that have withstood centuries of monsoon rains and changing dynasties.\n\nAt the very threshold of the temple entrance, etched permanently into the black stone steps, lies a deeply moving inscription: 'Seveche Thayi Tatpar Hiroji Indulkar' (Always in service, Hiroji Indulkar). When Shivaji Maharaj, deeply impressed by the construction of Raigad, asked his chief architect Hiroji what royal reward he desired, Hiroji humbly asked only for his name to be carved where the dust of the King's feet and the footsteps of visiting devotees would forever touch it.",
        "Hindi": "रायगड किले के शांत और पावन पठार पर स्थित 'जगदीश्वर मंदिर' भगवान शिव को समर्पित एक अद्वितीय वास्तुशिल्प और आध्यात्मिक केंद्र है। काले पत्थरों से निर्मित यह भव्य मंदिर मराठा और भारतीय स्थापत्य कला के अद्भुत सामंजस्य का प्रतीक है।\n\nछत्रपति शिवाजी महाराज प्रतिदिन प्रातःकाल इसी पावन गर्भगृह में ध्यान और आराधना करते थे। मंदिर के विशाल नंदी मंडप और भव्य स्तंभ सदियों के इतिहास और मराठा गरिमा के साक्षी हैं।\n\nइस मंदिर की सबसे भावुक कर देने वाली विशेषता इसके मुख्य प्रवेश द्वार की सीढ़ी पर खुदा हुआ शिलालेख है: 'सेवेचे ठायी तत्पर हिरोजी इंदुलकर'। जब शिवाजी महाराज ने संपूर्ण रायगड के निर्माण से प्रसन्न होकर मुख्य वास्तुकार हिरोजी इंदुलकर से मनचाहा पुरस्कार मांगने को कहा, तो हिरोजी ने केवल यह वरदान मांगा कि उनका नाम मंदिर की पहली पायरी पर अंकित किया जाए, ताकि भगवान के दर्शन करने आने वाले शिवभक्तों और महाराज के चरणों की धूलि उनके नाम को हमेशा स्पर्श करती रहे।",
        "Marathi": "दुर्गराज रायगडाच्या मध्यभागी शांत आणि प्रसन्न वातावरणात उभे असलेले 'जगदीश्वर मंदिर' हे भगवान शंकराचे अत्यंत पवित्र आणि भव्य शिवमंदिर आहे. काळ्या पाषाणात कोरलेले हे मंदिर मराठा आणि भारतीय स्थापत्यकलेचा एक अप्रतिम आविष्कार आहे.\n\nछत्रपती शिवाजी महाराज रोज सकाळी याच मंदिरात येऊन शंकराचे दर्शन घेत आणि स्वराज्यकार्यासाठी आत्मिक बळ मिळवत असत. मंदिरातील नंदी, भव्य खांब आणि प्रशस्त गाभारा शिवकालीन भक्ती आणि वास्तुरचनेची साक्ष देतात.\n\nया मंदिराच्या पायरीवर कोरलेला शिलालेख आजही प्रत्येक शिवप्रेमीचे डोळे पाणावतो: 'सेवेचे ठायी तत्पर हिरोजी इंदुलकर'. संपूर्ण रायगडची अभेद्य बांधणी पूर्ण झाल्यावर महाराजांनी मुख्य वास्तुविशारद हिरोजी इंदुलकर यांना विचारले, 'हिरोजी, तुमच्या या अफाट सेवेबद्दल आम्ही तुम्हाला काय बक्षीस द्यावे?' तेव्हा हिरोजींनी जहागीर किंवा सोने न मागता नम्रपणे मागितले, 'महाराज, मला फक्त जगदीश्वराच्या पायरीवर माझे नाव कोरण्याची परवानगी द्या, जेणेकरून मंदिरात येणाऱ्या प्रत्येक शिवभक्ताच्या आणि आपल्या चरणांची धूळ माझ्या नावावर सतत पडत राहील!'"
    },
    "Raigad Fort Maha Darwaja": {
        "English": "Maha Darwaja is the colossal, impregnable main portal that guarded the sovereign heart of the Maratha Empire. Strategically embedded between two monumental natural bastions towering over 65 feet high, the entrance is designed with military mastery: its approach curves along a narrow, steep zigzag pathway that prevented enemy artillery from aiming direct shots.\n\nThe massive wooden gateway was fitted with heavy iron spikes designed to impale charging war elephants. Deep guard rooms and concealed embrasures allowed Maratha musketeers and archers to rain arrows and molten defense upon any invader. Passing through this gateway, one can still hear the echoes of royal trumpets (Tutari) welcoming the legendary King to his capital.",
        "Hindi": "महा दरवाजा रायगड किले का वह विशाल और अजेय प्रवेश द्वार है, जो मराठा साम्राज्य के स्वाभिमान की रक्षा करता था। 65 फीट से अधिक ऊंचे दो विशाल प्राकृतिक बुरुजों के बीच स्थापित इस दरवाजे का निर्माण अद्वितीय युद्धनीति के साथ किया गया था। इसका घुमावदार और संकरा मार्ग तोपों के सीधे निशाने से पूरी तरह सुरक्षित था।\n\nलकड़ी के भारी कपाटों पर लगे नुकीले लोहे के शूल युद्ध के हाथियों के प्रहार को निष्फल कर देते थे। दरवाजे के दोनों ओर बने पहरेदारों के गुप्त कक्ष और झरोखे किसी भी शत्रु पर तीर और गोलियां बरसाने के लिए अचूक थे। इस दरवाजे से गुजरते हुए आज भी मराठा गौरव और तुतारी की गूंज का जीवंत अहसास होता है।",
        "Marathi": "महादरवाजा हा रायगडचा मुख्य, भव्य आणि अजेय असा महाद्वार आहे. ६५ फुटांपेक्षा उंच असणाऱ्या दोन प्रचंड बुरुजांच्या कपारीत बांधलेला हा दरवाजा अशा युद्धनीतीने रचला आहे की दुरून तोफेचा गोळा त्यावर थेट आदळूच शकत नाही. गडाचा मार्ग नागमोडी आणि अरुंद असल्याने शत्रूची मोठी फौज किंवा तोफा इथे सहज पोहोचू शकत नसत.\n\nदाराच्या मजबूत सागवानी फळ्यांवर लोखंडी अणकुचीदार खिळे बसवलेले होते, ज्यामुळे हत्तींनी धडक दिली तरी दरवाजा अभेद्य राहत असे. दरवाज्याच्या दोन्ही बाजूंना पहारेकऱ्यांसाठी 'पहारेकऱ्यांच्या देवड्या' आणि तोफा-बंदुका डागण्यासाठी गुप्त झरोके आहेत. या महादरवाज्यातून प्रवेश करताना आजही शिवकालीन तुतारीचे ललकार आणि मावळ्यांचा जयघोष कानात गुंजल्यासारखा वाटतो."
    },
    "Raigad Fort Shivaji Maharaj Samadhi": {
        "English": "The Samadhi of Chhatrapati Shivaji Maharaj at Raigad Fort is the supreme sanctum and eternal pilgrimage site of Indian history. Located in the peaceful precinct adjacent to the Jagadishwar Temple, this sacred memorial stands as a timeless beacon honoring the greatest warrior-king who championed Hindavi Swarajya, universal welfare, religious tolerance, and unmatched military genius.\n\nFollowing his passing on April 3, 1680, this memorial was raised upon the sacred grounds where his final rites were solemnized. Over the centuries, the great social reformer Lokmanya Bal Gangadhar Tilak and social pioneer Mahatma Jyotirao Phule revived and restored the sanctity of this site, igniting the national freedom struggle. Standing before the Samadhi, amidst the silent whispers of the Sahyadri winds, one feels the profound spirit of an immortal sovereign who dedicated every breath to his people.",
        "Hindi": "छत्रपति शिवाजी महाराज की पावन समाधि रायगड किले का सबसे पूजनीय और राष्ट्रीय तीर्थस्थल है। जगदीश्वर मंदिर के समीप स्थित यह पवित्र स्मारक उस महान युगपुरुष को नमन करता है, जिन्होंने न्याय, समानता, सामाजिक समरसता और अदम्य शौर्य के साथ हिंदवी स्वराज्य की स्थापना की।\n\n3 अप्रैल 1680 को महाराज के महाप्रयाण के बाद इसी पावन धरा पर उनका अंतिम संस्कार संपन्न हुआ था। बाद में राष्ट्रपिता महात्मा ज्योतिराव फुले और लोकमान्य बालगंगाधर तिलक ने इस पावन समाधि का जीर्णोद्धार कर इसे स्वतंत्रता संग्राम की अमर प्रेरणा बनाया। इस समाधि के समक्ष नतमस्तक होते ही सह्याद्रि की हवाओं में उस अद्वितीय जननायक की अमर गाथा का जीवंत स्पंदन महसूस होता है।",
        "Marathi": "छत्रपती शिवाजी महाराजांची समाधी हे रायगडावरील परमपवित्र, वंदनीय आणि अवघ्या महाराष्ट्राचे आराध्य तीर्थक्षेत्र आहे. जगदीश्वर मंदिराच्या प्रांगणात शांततेत उभे असलेले हे समाधी स्मारक हिंदवी स्वराज्याचे संस्थापक, रयतेचे राजे आणि अजेय योद्धा छत्रपती शिवाजी महाराजांच्या पावन स्मृतींना अखंड तेवत ठेवणारे प्रेरणास्थान आहे.\n\n३ एप्रिल १६८० रोजी शिवछत्रपतींच्या महाप्रयाणानंतर याच पवित्र भूमीवर त्यांचा अंत्यसंस्कार पार पडला. पुढे क्रांतीसूर्य महात्मा जोतीराव फुले यांनी या समाधीचा शोध घेऊन पहिला शिवजयंती उत्सव सुरू केला आणि लोकमान्य टिळकांनी या समाधीचा जीर्णोद्धार करून संपूर्ण देशाला स्वातंत्र्यलढ्याची नवी चेतना दिली. या समाधीसमोर नतमस्तक होताना सह्याद्रीच्या वाऱ्यातून रयतेच्या राजाच्या अथांग त्यागाची आणि पराक्रमाची अनुभूती येते."
    },
    "Raigad Fort Takmak Tok": {
        "English": "Takmak Tok is an awe-inspiring, knife-edge cliff at the northernmost rim of Raigad Fort, plunging sheer into a breathtaking 1,200-foot abyss below. With fierce winds whipping across the mountain ridge, it offers sweeping, dizzying panoramic views of the dense Konkan jungles and the Sahyadri mountains.\n\nHistorically, this dizzying precipice served as the solemn point of royal justice. Traitors and conspirators who committed treason against Swarajya were blindfolded, tied, and hurled into the deep gorge. The terrifying sheer drop of Takmak Tok served as an unyielding warning that while the King's compassion was boundless for the righteous, treason against the motherland was met with absolute, uncompromising justice.",
        "Hindi": "टकमक टोक रायगड किले के उत्तरी छोर पर स्थित एक अत्यंत रोमांचक, संकीर्ण और सीधी खड़ी चट्टान है, जो 1,200 फीट से अधिक गहरी खाई में उतरती है। यहाँ चलने वाली तेज तूफानी हवाएं और नीचे फैले घने कोंकण के जंगल इस स्थान को अत्यंत विस्मयकारी और अद्भूत बनाते हैं।\n\nऐतिहासिक रूप से, यह स्थान स्वराज्य में न्याय और दंड का प्रतीक था। स्वराज्यद्रोह और विश्वासघात करने वाले अपराधियों को आंखों पर पट्टी बांधकर इसी गहरी खाई में फेंकने का कठोर दंड दिया जाता था। टकमक टोक की यह भयानक खाई इस बात का जीवंत प्रमाण थी कि छत्रपति शिवाजी महाराज की रयत के लिए जहां असीम करुणा थी, वहीं देशद्रोहियों के लिए न्याय अटूट और अटल था।",
        "Marathi": "टकमक टोक हा रायगडच्या उत्तर टोकावरील अंगावर काटा आणणारा, अत्यंत अरुंद आणि खोल अथांग दरीत कोसळणारा सरळ कडा आहे. या टोकावर उभे राहिले की खाली १,२०० फुटांपेक्षा खोल दरी आणि समोर सह्याद्रीच्या रौद्रसुंदर पर्वतशिखरांचे विहंगम दृश्य डोळ्यांचे पारणे फेडते.\n\nशिवकाळात हे टोक न्यायनिवाडा आणि कठोर शिक्षेचे ठिकाण होते. स्वराज्याशी गद्दारी करणाऱ्या आणि देशद्रोह करणाऱ्या गुन्हेगारांचे डोळे बांधून त्यांना या टोकावरून खोल दरीत लोटून देण्याची शिक्षा दिली जात असे. टकमक टोक हा या गोष्टीचा संदेश होता की, शिवछत्रपतींचे हृदय रयतेसाठी जेवढे मायाळू आणि विशाल होते, तेवढाच स्वराज्याशी द्रोह करणाऱ्यांसाठी त्यांचा न्याय अत्यंत कठोर आणि निःपक्षपाती होता."
    },
    "Raigad Fort": {
        "English": "Raigad Fort, soaring proudly at 820 meters amidst the Sahyadri mountain ranges, was crowned by Chhatrapati Shivaji Maharaj as the capital of the Maratha Empire. It was here, on June 6, 1674, that Shivaji Maharaj was coronated as Chhatrapati in a grand ceremony that announced the sovereign emergence of Hindavi Swarajya to the world.\n\nThe fort encompasses royal palaces, the Raj Sabha (Queen and King's assembly with extraordinary acoustic design where a whisper at the entrance carries to the throne), the two-tiered Bazar Peth (marketplace designed for horse-mounted shopping), the Queen's Quarters (Ranivasa), and the sacred Ganga Sagar Lake. Raigad stands as an enduring monument to Indian sovereignty, statecraft, and timeless valor.",
        "Hindi": "सह्याद्रि की पर्वत श्रृंखलाओं में 820 मीटर की ऊंचाई पर स्थित रायगड किला छत्रपति शिवाजी महाराज द्वारा स्थापित मराठा साम्राज्य की गौरवशाली राजधानी था। 6 जून 1674 को इसी पावन धरा पर छत्रपति शिवाजी महाराज का ऐतिहासिक राज्याभिषेक हुआ, जिसने समूचे विश्व में हिंदवी स्वराज्य की संप्रभुता का शंखनाद किया।\n\nकिले पर स्थित 'नगारखाना' और 'राजसभा' की वास्तुकला इतनी अद्भुत है कि प्रवेश द्वार की धीमी फुसफुसाहट भी सीधे सिंहासन तक स्पष्ट सुनाई देती है। इसके अलावा 40 से अधिक दुकानों वाली विशाल 'बाजारपेठ', 'राणीवसा', और पवित्र 'गंगासागर तलाव' मराठा स्थापत्य कला और दूरदर्शिता के अमर प्रमाण हैं।",
        "Marathi": "सह्याद्रीच्या कुशीत ८२० मीटर उंचीवर अभेद्यपणे उभा असलेला दुर्गराज रायगड हा छत्रपती शिवाजी महाराजांच्या हिंदवी स्वराज्याची राजधानी आणि मराठा अस्मितेचे सर्वोच्च तीर्थक्षेत्र आहे. ६ जून १६७४ रोजी याच गडावर छत्रपती शिवरायांचा सुवर्ण राज्याभिषेक झाला आणि स्वातंत्र्याचा भगवा ध्वज दिमाखाने फडकला.\n\nरायगडावरील नगारखाना आणि राजदरबाराची वास्तुकला अशी विलक्षण आहे की नगारखान्यात केलेला अगदी हलका आवाजही थेट सिंहासनापर्यंत स्पष्ट ऐकू येतो. याशिवाय घोड्यावरून फिरून खरेदी करता येणारी दुमजली 'बाजारपेठ', 'राणीवसा', 'होळीचा माळ', आणि 'गंगासागर तलाव' गडाचे अतुलनीय वैभव प्रकट करतात."
    }
}

def query_openrouter_llm(prompt: str, max_tokens: int = 650) -> str:
    """Helper to query OpenRouter using multi-model auto fallback for rich storytelling."""
    candidate_models = [
        "openrouter/auto",
        "deepseek/deepseek-chat",
        "google/gemini-2.5-flash"
    ]
    for model_name in candidate_models:
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://raigad-place-classifier.com",
                    "X-Title": "Raigad Place Classifier",
                },
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            content = completion.choices[0].message.content.strip()
            if content and len(content) > 30:
                return content
        except Exception as e:
            print(f"[HistTwin AI] LLM model {model_name} notice: {e}")
            continue
    return None

def get_fallback_story(place_name: str, language: str = "English", user_question: str = None) -> str:
    """Provides rich, guaranteed local knowledge fallback with smart landmark recognition."""
    combined_query = f"{place_name or ''} {user_question or ''}".lower()
    matched_key = None
    
    if any(w in combined_query for w in ["hirkani", "buruj", "milkmaid", "वळिवंडे", "हिरकणी"]):
        matched_key = "Raigad Fort Hirkani Buruj"
    elif any(w in combined_query for w in ["jagadishwar", "jagdishwar", "temple", "shiva", "hiroji", "मंदि", "जगदीश्वर"]):
        matched_key = "Raigad Fort Jagadishwar Temple"
    elif any(w in combined_query for w in ["maha darwaja", "darwaja", "gate", "door", "दरवाजा", "महादरवाजा"]):
        matched_key = "Raigad Fort Maha Darwaja"
    elif any(w in combined_query for w in ["samadhi", "tomb", "memorial", "समाधी"]):
        matched_key = "Raigad Fort Shivaji Maharaj Samadhi"
    elif any(w in combined_query for w in ["takmak", "tok", "cliff", "valley", "टकमक"]):
        matched_key = "Raigad Fort Takmak Tok"
    else:
        matched_key = "Raigad Fort"

    base_story = RAIGAD_KNOWLEDGE_BASE[matched_key].get(language, RAIGAD_KNOWLEDGE_BASE[matched_key]["English"])
    if user_question:
        if language == "Marathi":
            return f"**{user_question}**\n\n{base_story}"
        elif language == "Hindi":
            return f"**{user_question}**\n\n{base_story}"
        else:
            return f"**Regarding: {user_question}**\n\n{base_story}"
    return base_story

def get_place_details(place_name, language="English", user_question=None):
    """
    Fetches immersive, story-based details about the place with instant in-memory cache and guaranteed fallback.
    """
    if not place_name or place_name == "Unknown" or place_name == "Not a recognizable place":
        return "No specific information available for this place."
    
    cache_key = f"{place_name}_{language}_{user_question or 'overview'}"
    if cache_key in PLACE_CACHE:
        print(f"[HistTwin AI] Instant cache hit for {cache_key}")
        return PLACE_CACHE[cache_key]

    try:
        if user_question:
            prompt = f"You are HistTwin AI, the master historical chronicler and virtual storyteller of Raigad district, Maharashtra. Answer the following question about {place_name}: '{user_question}'. Provide a detailed, immersive, captivating historical story in 3-4 rich paragraphs detailing the legends, strategic context, emotional heroics of Chhatrapati Shivaji Maharaj's era, and cultural importance. Provide the response entirely in {language}."
        else:
            prompt = f"You are HistTwin AI, the master historical chronicler of Raigad district, Maharashtra. Provide a detailed, captivating, multi-paragraph story-based historical chronicle of {place_name}. Describe its dramatic geographical setting in the Sahyadri mountains, the heroic legends and folklore, the architectural and military genius of the Maratha Empire under Chhatrapati Shivaji Maharaj, and why it inspires visitors today. Write an engaging narrative in 3-4 rich paragraphs entirely in {language}."
        
        answer = query_openrouter_llm(prompt, max_tokens=650)
        if not answer:
            answer = get_fallback_story(place_name, language, user_question)
            
        PLACE_CACHE[cache_key] = answer
        return answer

    except Exception as e:
        print(f"Error fetching history: {e}")
        fallback = get_fallback_story(place_name, language, user_question)
        PLACE_CACHE[cache_key] = fallback
        return fallback

def general_chat(user_question, language="English"):
    """
    Handles general chat queries with deep, engaging storytelling.
    """
    if not user_question:
        return "How can I help you today?"
        
    cache_key = f"{user_question.strip().lower()}_{language}"
    if cache_key in GENERAL_CHAT_CACHE:
        return GENERAL_CHAT_CACHE[cache_key]

    try:
        prompt = f"You are HistTwin AI, a captivating, deeply knowledgeable AI historian and guide for tourists visiting Raigad district, Maharashtra. Provide a rich, inspiring, and helpful story-based response in 2-3 engaging paragraphs for: '{user_question}'. Provide the response entirely in {language}."
        
        answer = query_openrouter_llm(prompt, max_tokens=650)
        if not answer:
            answer = get_fallback_story("Raigad Fort", language, user_question)
            
        GENERAL_CHAT_CACHE[cache_key] = answer
        return answer

    except Exception as e:
        print(f"Error in general chat: {e}")
        fallback = get_fallback_story("Raigad Fort", language, user_question)
        GENERAL_CHAT_CACHE[cache_key] = fallback
        return fallback

def generate_video_prompt(story_text):
    """Summarizes the story into a visual prompt for Sora."""
    if not story_text or len(story_text) < 10:
        return "A beautiful sweeping cinematic shot of a historical place in Raigad, Maharashtra, India."
    
    try:
        prompt = f"Convert the following historical story into a single, highly visual, descriptive prompt suitable for a text-to-video AI model like Sora. The prompt should focus on dynamic visual elements, scenery, and atmosphere. Keep it concise (under 50 words).\n\nStory: {story_text}"
        
        completion = client.chat.completions.create(
          extra_headers={
            "HTTP-Referer": "https://raigad-place-classifier.com",
            "X-Title": "Raigad Place Classifier",
          },
          model="google/gemini-2.5-flash",
          messages=[
            {"role": "user", "content": prompt}
          ],
          max_tokens=300
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating video prompt: {e}")
        return "A cinematic aerial view of a historical fort in Raigad, Maharashtra, showing beautiful architecture and lush green surroundings."

VEO_API_KEY = os.getenv("VEO_API_KEY") or os.getenv("GEMINI_API_KEY", "")

import time

def generate_sora_video(prompt):
    """Calls the Google AI Veo API to generate a video based on the visual prompt."""
    model = "models/veo-3.1-generate-preview"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:predictLongRunning?key={VEO_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "instances": [{"prompt": prompt}]
    }
    
    try:
        print(f"Calling Veo API with prompt: {prompt[:50]}...")
        response = requests.post(url, headers=headers, json=data, timeout=4)
        
        if response.status_code == 200:
            result = response.json()
            if "name" in result:
                operation_name = result["name"]
                operation_url = f"https://generativelanguage.googleapis.com/v1beta/{operation_name}?key={VEO_API_KEY}"
                
                # Poll max 2 times (10s max) to prevent blocking HTTP response
                for _ in range(2):
                    time.sleep(3)
                    poll_response = requests.get(operation_url, timeout=3)
                    if poll_response.status_code == 200:
                        poll_data = poll_response.json()
                        if poll_data.get("done"):
                            if "response" in poll_data:
                                response_obj = poll_data["response"]
                                if "videoUri" in response_obj:
                                    return response_obj["videoUri"]
                                elif "videoBytes" in response_obj:
                                    return f"data:video/mp4;base64,{response_obj['videoBytes']}"
                            return None
            return None
        else:
            print(f"Veo API status {response.status_code} (non-blocking fallback)")
            return None
    except Exception as e:
        print(f"Veo API non-blocking fallback: {e}")
        return None


def predict_image(image_path):
    model = load_raigad_model()
    if model is None:
        return None, "Model not found"

    try:
        # Preprocess image
        test_image = image.load_img(image_path, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        # ResNet50 preprocess_input layer in the model handles zero-centering and BGR conversion
        # from 0-255 RGB inputs.
        test_image = np.expand_dims(test_image, axis = 0)
        
        # Predict
        result = model.predict(test_image)
        
        # Get class
        class_names = get_class_names()
        if not class_names:
            return None, "Class names not found"

        predicted_index = np.argmax(result[0])
        prediction = class_names[predicted_index]
        confidence = float(result[0][predicted_index] * 100)
        
        # If the model explicitly predicts "Unknown", return that
        if prediction == "Unknown":
            return "Not a recognizable place", confidence

        # Optional: Keep a small threshold just in case, but much lower
        if confidence < 60.0:
             return "Not a recognizable place", confidence
        
        return prediction, confidence
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None, str(e)
