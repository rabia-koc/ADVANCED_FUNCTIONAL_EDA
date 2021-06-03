# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime,  timedelta


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Veri setiyle ilgili yapısal bilgilere erişmek için bir genelleştirilmiş fonksiyon oluşturabiliriz.
# info()'yu print içinde yazmıyoruz çünkü eğer bir elemana sahip değilse none döndürür.!!
def check_df(dataframe, head=5):
    print(" - shape - ".upper().center(50, "*"))
    print(dataframe.shape)
    print(" - types - ".upper().center(50, "*"))
    print(dataframe.dtypes)
    print(" - head - ".upper().center(50, "*"))
    print(dataframe.head(head))
    print(" - tail - ".upper().center(50, "*"))
    print(dataframe.tail(head))
    print(" - info - ".upper().center(50, "*"))
    dataframe.info()
    print(" - columns - ".upper().center(50, "*"))
    print(dataframe.columns)
    print(" - values - ".upper().center(50, "*"))
    print(dataframe.values)
    print(" - index - ".upper().center(50, "*"))
    print(dataframe.index)
    print(" - nunique - ".upper().center(50, "*"))
    print(dataframe.nunique())
    print(" - na - ".upper().center(50, "*"))
    print(dataframe.isnull().sum())
    print(" - quantiles - ".upper().center(50, "*"))
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print(" - describe - ".upper().center(50, "*"))
    print(dataframe.describe().T)
    print()

# Her seferinde veri setini okumak yerine tek seferlik fonksiyon yazıp istediğimiz zaman çağırabiliriz.
def load_churn():
    return pd.read_csv('HAFTA_02/churn.csv', index_col=0)

df = load_churn()

check_df(df)   # check_df fonksiyonuyla dataframe çağırdığımızda bütün bilgilere erişmiş olacağız.

# Değişkenleri inceleyelim.
# Kaç unique Names vardır?
df["Names"].nunique()

# Account_Manager Frekansları nelerdir?
df["Account_Manager"].value_counts()

# Churn'e göre değişkenlerin sayıları nelerdir?
df.groupby("Churn").count()

# Churn kırılımında Age ortalamaları nelerdir?
df.groupby("Churn").agg({"Age":"mean"})

# Churn kırılımında Total_Purchase toplamı nelerdir? (int şekilde)
df.groupby("Churn").agg({"Total_Purchase":"sum"}).astype(int)

# Veri görselleştirmesi
sns.pairplot(df, hue = "Churn")
plt.show()

sns.pairplot(df, vars = ["Age", "Total_Purchase","Years","Num_Sites"], hue = "Churn", kind = "reg")
plt.show()

# Müşterini alışverişe başladığı tarihi bulmak istersek,
# Önce Years değişkeni üzerinden her yılı 365 ile çarparız ve günlere erişiriz.
# Sonra günler üzerinden de bu günden days çıkartarak müşterinin alışverişe başladığı tarihe erişiriz.
# Eriştiğimiz tarihte tarih ve saat bilgisi bulunduğundan dolayı sadece tarih bilgisini çekeriz.
def customers_hire_date(dataframe):
    dataframe = dataframe["Years"].apply(lambda row: row * 365).round().astype(int)
    dataframe = dataframe.apply(lambda x: datetime.today() - timedelta(days=x))
    dataframe["Date"] = pd.to_datetime(dataframe).apply(lambda x: x.strftime('%d %B %Y'))
    return dataframe["Date"]
customers_hire_date(df)

# cat_summary fonksiyonu ile kategorik değişkenler bakımından analiz gerçekleştiririz.
# Plot_type ön tanımlı değeri "count_plot" 'tur.
# Girilen değişkenin sınıf frekanslarını ve sınıf frekanslarını yüzdelik olarak verir.
def cat_summary(dataframe, col_name, plot=False, plot_type="count_plot"):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("uzunluk bilgisi".upper().center(50, "*"))
    print(f'{col_name}: {len(col_name)}')
    if plot:
        if plot_type=="count_plot":
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.xlabel(col_name)
            plt.title(col_name)
            plt.show()
        elif plot_type=="bar_plot":
            # 2 türlü çizdirdim birisini kapattım.
            #sns.barplot(x=dataframe[col_name], y=df[col_name].index, data=dataframe);
            df[col_name].value_counts().plot.barh().set_title(F'"{col_name}" Değişkeninin Sınıf Frekansları')
            plt.show()
        else:
            print("Grafik türü hatalı! Lütfen tekrar deneyiniz.")


# Girilen sayısal değişkenleri çeyrekliklerine göre betimsel analinizi bizlere verir.
# Plot_type ön tanımlı değeri "box_plot" 'tur
def num_summary(dataframe, numerical_col, plot=False, plot_type="box_plot"):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        if plot_type == "hist":
            dataframe[numerical_col].hist(bins=30)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()
        elif plot_type == "box_plot":
            sns.boxplot(x=dataframe[numerical_col])
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            plt.show()
        else:
            print("doğru grafik türü değil")

# veri setindeki her nümerik değişken nümerik değildir.
# Sayısal olan fakat 10'dan az sayıda sınıfa sahip olan değişkenleri nasıl yakalarız?
# nümerik görünümlü aslında kategorik değişkenleri yakalamak için
# bir değişkenin eğer eşsiz sınıf sayısı 10'dan küçükse ve tipi object'ten farklı ise numeric görünümlü kategorik değişkenlere erişiriz.

# Peki kategorik olan fakat ölçülemeyecek kadar fazla sayıda sınıfa sahip olan değişkenleri yakalamak istiyoruz diyelim.
# değişken ölçülemez, cardinalitesi yüksek ise kardinaldir.

# Burada değindiklerimizi genelleştirilmiş fonksiyon yapısıyla ile eşik değerlerini kullanıcı tarafından girilecek şekilde oluşturalım.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Girilen veri setinde bulunan kategorik, numeric, kategorik görünümlü kardinal değişkenlerin isimlerini verir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimlerinin incelenmesi istenilen dataframe
        cat_th: int, optional
                numeric ama kategorik değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik ama kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişkenlerin listesi
        num_cols: list
                Numerik değişkenlerin listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişkenlerin listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("tips")
        print(grab_col_names(df))

    Notes
    ------
    Not-1:
        cat_cols + num_cols + cat_but_car = toplam değişken sayısını ifade eder.
        num_but_cat cat_cols'un içerisinde mevcuttur.
    Not-2:
         Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    """

    # categorik kolonlar ve categorik görünümlü kardinaller,
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # cat_cols var bir de numerik gibi davranan cat'ler var. Hepsini bir araya getirelim.
    cat_cols = cat_cols + num_but_cat
    # kategorikler içerisinden kardinalleri çıkaralım
    # cat_cols içinde cardinal olanlar var onları ayıklamak gerek.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numeric kolonlar
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # numeric görünümlü kategorik değişkenleri ayıklayalım sadece numeric değişkenlerimiz kalsın.
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(F"Observations: {dataframe.shape[0]}")
    print(F"Variables: {dataframe.shape[1]}")
    print(F'cat_cols: {len(cat_cols)}')
    print(F'num_cols: {len(num_cols)}')
    print(F'cat_but_car: {len(cat_but_car)}')
    print(F'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# kategorik değişkenler için yazdığımız fonksiyonu çağıralım.
for col in cat_cols:
    print({f"{col}"})
    print(cat_summary(df, col, plot=True))

# numeric fonksiyonlar için yazdığımız fonksiyonu çağıralım.
for col in num_cols:
    print({f"{col}"})
    print(num_summary(df, col, plot=True, plot_type="hist"))








