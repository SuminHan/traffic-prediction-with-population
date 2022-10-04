import fiona
import numpy as np
import os
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
import geopandas as gpd
from cartoframes.viz import Layer, Map
from tqdm import notebook
from shapely.geometry import Point, Polygon


b00 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.0182121254887,37.51933304303243],[127.01586968695781,37.51517975496106],[127.01909364851083,37.5161793919259],[127.0182121254887,37.51933304303243]]]}}]}
b01 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.01819009228703,37.521765754450946],[127.01857445402112,37.52073855136515],[127.01893606978292,37.5192698589646],[127.01937688087015,37.51790927253251],[127.01981768310874,37.51657571425177],[127.02788400261049,37.51992581772709],[127.02823705048216,37.52623275182653],[127.02825991763989,37.52687245780315],[127.01819009228703,37.521765754450946]]]}}]}
b02 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02875779065677,37.5271336281926],[127.02870059713729,37.52549381833357],[127.02833652219164,37.52003383039311],[127.03347290319287,37.521690331744225],[127.03288537888584,37.523312301554235],[127.03287450919461,37.524321427318654],[127.03310166150192,37.5263576288713],[127.03347610776825,37.52886230851011],[127.0334082720737,37.528961437901906],[127.0324126170544,37.52877250500262],[127.03134900445146,37.52838536137907],[127.03053430761945,37.52803418412891],[127.02875779065677,37.5271336281926]]]}}]}
b03 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.0340418308989,37.52898828703212],[127.03380348647542,37.52729447214369],[127.03339488216355,37.52430326128076],[127.03342847431495,37.52352838968916],[127.03357525835376,37.52290665630375],[127.03401597219212,37.52187037713717],[127.03896023559561,37.52341857416347],[127.0405010288327,37.52766177602965],[127.03676816328604,37.52861805903218],[127.03565959062178,37.52888869946232],[127.03479982016246,37.5289790560982],[127.0340418308989,37.52898828703212]]]}}]}
b04 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04077249010344,37.527562572422454],[127.03940146074427,37.52349050799034],[127.04698141436697,37.52440675845691],[127.04619044604566,37.525893724082906],[127.0452517992529,37.52642567866641],[127.04077249010344,37.527562572422454]]]}}]}
b10 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.01574519693489,37.51486442028455],[127.01563177669719,37.51338678885972],[127.01592546018041,37.511494640778324],[127.01637752577903,37.50983673071097],[127.0209020858179,37.51120554357206],[127.01983963340197,37.514034879931806],[127.01934242495219,37.51589103027719],[127.01574519693489,37.51486442028455]]]}}]}
b11 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.0199080311977,37.516035096887364],[127.0206312650664,37.51351216198987],[127.02158077161867,37.5113676023141],[127.03044943946036,37.5140686615024],[127.02796297791774,37.51936717713801],[127.0199080311977,37.516035096887364]]]}}]}
b12 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02841550107182,37.519493209573724],[127.0337102411919,37.521203722727684],[127.0349306103584,37.518320158319085],[127.0350660474849,37.517671395789804],[127.03506507783152,37.515599087940345],[127.03090196450955,37.51426676484287],[127.02841550107182,37.519493209573724]]]}}]}
b13 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03427589863323,37.52131168026616],[127.03554147765085,37.518392056343785],[127.03567595738309,37.51574306559802],[127.04108365456008,37.51734509474567],[127.03898256993031,37.52285994522349],[127.03427589863323,37.52131168026616]]]}}]}
b14 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03941251730694,37.52300396294305],[127.04728655997336,37.52393811545732],[127.0499078114228,37.519035581929614],[127.04142306161062,37.517453096610204],[127.03941251730694,37.52300396294305]]]}}]}
b15 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04776174537598,37.52402802448834],[127.05042815560128,37.51903536091386],[127.05655999669499,37.52013180742428],[127.05407466113846,37.52471006460342],[127.04776174537598,37.52402802448834]]]}}]}
b16 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05502548263246,37.525466467275265],[127.05493459703906,37.5249439283294],[127.05746509442525,37.52031157111322],[127.0613575657661,37.52169715742459],[127.06086047169327,37.522490296919266],[127.0555685237632,37.52550225310707],[127.05502548263246,37.525466467275265]]]}}]}
b20 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.01660364687766,37.509440257100515],[127.02110556592297,37.51080906544212],[127.02404434880815,37.50466365022092],[127.02004039030572,37.503402989658234],[127.0188415406966,37.503222985836295],[127.01660364687766,37.509440257100515]]]}}]}
b21 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02171641097856,37.51106123596071],[127.03058504744388,37.51376228511834],[127.03352312031402,37.50750851685436],[127.02476820900971,37.50478964141806],[127.02171641097856,37.51106123596071]]]}}]}
b22 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03108280032055,37.51392433589839],[127.03513272607593,37.51513054589753],[127.0351091412489,37.51307626396121],[127.03538016981595,37.51215715885594],[127.03589996574527,37.51111183702985],[127.03807042721972,37.50878656568304],[127.03404340094751,37.50754440825308],[127.03108280032055,37.51392433589839]]]}}]}
b23 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03563051499464,37.515328617610685],[127.0411513261105,37.51698466981188],[127.04363606276723,37.510460509322414],[127.0387038869162,37.508948540673],[127.03712126456442,37.51060689705846],[127.03617183842786,37.51197671822886],[127.03567481654126,37.51334639568534],[127.03563051499464,37.515328617610685]]]}}]}
b24 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04153598776558,37.517110675652376],[127.05002065306027,37.51862107274264],[127.0529579049629,37.51323179082524],[127.04399805283185,37.510550475487406],[127.04153598776558,37.517110675652376]]]}}]}
b25 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05058627792289,37.51867489203702],[127.05345568973179,37.5133757272347],[127.05970015135972,37.51438186051519],[127.05674070205042,37.519753299507656],[127.05058627792289,37.51867489203702]]]}}]}
b26 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05757796539483,37.519987155090135],[127.0605826230875,37.514633692246456],[127.06653315852107,37.515603574249695],[127.06606032327623,37.518144669203124],[127.06508759598312,37.51823530956112],[127.0621031572893,37.52047140397869],[127.06142524709745,37.5214628615452],[127.05757796539483,37.519987155090135]]]}}]}
b30 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.01886407798358,37.50289862020725],[127.01994979463906,37.50295250218005],[127.0242703845964,37.504177061064574],[127.02714082903442,37.49812167662656],[127.02144075560318,37.49637489600287],[127.01886407798358,37.50289862020725]]]}}]}
b31 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02494904401391,37.50442920120388],[127.03370390356245,37.507112023225105],[127.03657369817118,37.50105641289674],[127.0278872754666,37.49831972417777],[127.02494904401391,37.50442920120388]]]}}]}
b32 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03420161900192,37.50727406106353],[127.03831906005026,37.50840806328984],[127.04105461537591,37.505632048830215],[127.04238760383373,37.502856481362095],[127.0370713617757,37.501182398597955],[127.03420161900192,37.50727406106353]]]}}]}
b33 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03888469295946,37.5086421396851],[127.0437715471851,37.51004599712935],[127.0441780976608,37.508982660348046],[127.04467539706197,37.50838781019151],[127.04790871602695,37.50618809104271],[127.04869948369286,37.50476417946049],[127.04295313526376,37.50296439775664],[127.04152981700322,37.50597426468638],[127.03888469295946,37.5086421396851]]]}}]}
b34 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04411094624697,37.51019003159454],[127.0532064517174,37.512817217273685],[127.05630129088003,37.50703133380693],[127.04910663205001,37.504764010907124],[127.04827084140524,37.50651230544506],[127.04707246293104,37.5072696332652],[127.0450826409835,37.50851379613277],[127.04456266975296,37.509018556078864],[127.04411094624697,37.51019003159454]]]}}]}
b35 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05363636873285,37.512961183484045],[127.059880798228,37.5139673073106],[127.06265919197394,37.50902836293616],[127.05666332039452,37.50717532148802],[127.05363636873285,37.512961183484045]]]}}]}
b36 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.06069532610526,37.51412907226177],[127.06646498702392,37.51526123125926],[127.06736544964974,37.5103232191693],[127.06345103619925,37.5091721009767],[127.06069532610526,37.51412907226177]]]}}]}
b40 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02153107806281,37.49587031582246],[127.02741207262156,37.49767111078818],[127.02976240405201,37.49298530663429],[127.02342939280851,37.49111263709892],[127.02153107806281,37.49587031582246]]]}}]}
b41 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02797759049814,37.49790523988544],[127.03689006981264,37.50047967070992],[127.03899119794188,37.49595594404284],[127.0302148363131,37.49327351407542],[127.02797759049814,37.49790523988544]]]}}]}
b42 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03731988478117,37.500623696665016],[127.04265867035532,37.502225679815865],[127.04433031522893,37.49862104036925],[127.03937573859638,37.49606393730093],[127.03731988478117,37.500623696665016]]]}}]}
b43 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04315638303791,37.50240570017672],[127.04894792010573,37.50418743335086],[127.05061935483243,37.50065478674386],[127.04487318917933,37.498710936032246],[127.04315638303791,37.50240570017672]]]}}]}
b44 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04940033930733,37.504241305342504],[127.05104919712096,37.500798762722646],[127.0582888556249,37.503174067405986],[127.05666257128783,37.5061842163591],[127.05618769098687,37.506364644205306],[127.04940033930733,37.504241305342504]]]}}]}
b45 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05716056364732,37.50665249904709],[127.05889974913893,37.503408025821116],[127.06471437176931,37.50520699414917],[127.06288491457838,37.50845160000973],[127.05716056364732,37.50665249904709]]]}}]}
b46 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.06375598620674,37.508667374190466],[127.06777580896532,37.50133095382615],[127.07076227604512,37.50223020831552],[127.06850214668005,37.50412365204496],[127.06735357242349,37.50969252283933],[127.06375598620674,37.508667374190466]]]}}]}
b50 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.023451940443,37.49089639082969],[127.02476252616746,37.48758041488502],[127.03127625041056,37.4895971276531],[127.02967185528439,37.49276908768699],[127.023451940443,37.49089639082969]]]}}]}
b51 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03032780648023,37.493003182986214],[127.03910412170953,37.49564956443603],[127.04054988929039,37.49251357277119],[127.03188695681355,37.48984924654762],[127.03032780648023,37.493003182986214]]]}}]}
b52 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03959047388277,37.49582960379409],[127.0445449674103,37.49826055710801],[127.04639719662488,37.49438551574199],[127.04112666768546,37.49265753454559],[127.03959047388277,37.49582960379409]]]}}]}
b53 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04504270158097,37.498512650143084],[127.05067568090595,37.50033040044979],[127.05257289055477,37.49647326452218],[127.04691742612782,37.49447541097446],[127.04504270158097,37.498512650143084]]]}}]}
b54 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.0511959813722,37.500456316944884],[127.05850349585735,37.502867619416485],[127.06055869001953,37.499010278665935],[127.05300263619414,37.49650911275857],[127.0511959813722,37.500456316944884]]]}}]}
b55 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05902386959288,37.50304756166588],[127.06107905657075,37.49920823211369],[127.06696134059904,37.5010971568031],[127.0648385763147,37.504972664688864],[127.05902386959288,37.50304756166588]]]}}]}
b60 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.02488680849528,37.48729206620322],[127.03141179951656,37.489263718840085],[127.03351295252361,37.48475810115205],[127.02625372182509,37.48375080768746],[127.02488680849528,37.48729206620322]]]}}]}
b61 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.03215817676458,37.48947975988993],[127.04073060419108,37.49212607749219],[127.04125013737118,37.490990627966895],[127.04136301464277,37.49063018551347],[127.04102142325036,37.48630546703668],[127.04113423531228,37.48583690355273],[127.03430444360319,37.48481193449999],[127.03215817676458,37.48947975988993]]]}}]}
b62 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04129605231574,37.49223400206549],[127.04192845239594,37.490738107093584],[127.0415865781903,37.48596288635086],[127.04513724115361,37.48648416794326],[127.04950311555082,37.488608798076655],[127.0465890931195,37.493835825983275],[127.04129605231574,37.49223400206549]]]}}]}
b63 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.04704152729501,37.49401584809158],[127.05283266062663,37.49601363526827],[127.05515859046209,37.49112911424336],[127.04993285065851,37.48869871801223],[127.04704152729501,37.49401584809158]]]}}]}
b64 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.05323986378649,37.496157613518854],[127.06077321414935,37.49858669559691],[127.06287337692544,37.49469324844703],[127.05554315626533,37.49127309524826],[127.05323986378649,37.496157613518854]]]}}]}
b65 = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},"features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[127.0615197537713,37.49878453054897],[127.06348438337365,37.495125404933724],[127.0693434904718,37.49668986255463],[127.06717575377907,37.50058346174497],[127.0615197537713,37.49878453054897]]]}}]}

b_list = dict()
b_list[(0,0)] = b00
b_list[(0,1)] = b01
b_list[(0,2)] = b02
b_list[(0,3)] = b03
b_list[(0,4)] = b04
b_list[(1,0)] = b10
b_list[(1,1)] = b11
b_list[(1,2)] = b12
b_list[(1,3)] = b13
b_list[(1,4)] = b14
b_list[(1,5)] = b15
b_list[(1,6)] = b16
b_list[(2,0)] = b20
b_list[(2,1)] = b21
b_list[(2,2)] = b22
b_list[(2,3)] = b23
b_list[(2,4)] = b24
b_list[(2,5)] = b25
b_list[(2,6)] = b26
b_list[(3,0)] = b30
b_list[(3,1)] = b31
b_list[(3,2)] = b32
b_list[(3,3)] = b33
b_list[(3,4)] = b34
b_list[(3,5)] = b35
b_list[(3,6)] = b36
b_list[(4,0)] = b40
b_list[(4,1)] = b41
b_list[(4,2)] = b42
b_list[(4,3)] = b43
b_list[(4,4)] = b44
b_list[(4,5)] = b45
b_list[(4,6)] = b46
b_list[(5,0)] = b50
b_list[(5,1)] = b51
b_list[(5,2)] = b52
b_list[(5,3)] = b53
b_list[(5,4)] = b54
b_list[(5,5)] = b55
b_list[(6,0)] = b60
b_list[(6,1)] = b61
b_list[(6,2)] = b62
b_list[(6,3)] = b63
b_list[(6,4)] = b64
b_list[(6,5)] = b65

ks = [{'y':b[0], 'x':b[1]} for b in list(b_list.keys())]
its = list(b_list.items())

block_gdf = gpd.GeoDataFrame(ks, geometry=[Polygon(it[1]['features'][0]['geometry']['coordinates'][0]) for it in its])
block_gdf.crs = 'epsg:4326'
block_gdf = block_gdf.to_crs('epsg:5181')

# print(block_gdf)

# lte_df = pd.read_hdf('../prepdata-1803-1808/gangnam/lte_cell_raw.h5')

# arr = np.zeros((len(bgdf), len(lte_gdf)))
# for i, gg in enumerate(bgdf.geometry):
#     for j, item in lte_gdf[lte_gdf.intersects(gg)].iterrows():
#         arr[i, j] = 1