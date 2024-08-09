# TRsemanticAnalysisTKNFST
![turkcedil2_KQLmW](https://github.com/user-attachments/assets/24b4fecf-0e68-4822-9a1b-33c9945df961)

Bu projede hazır bir LLM destekli sisteme yaklaşık 15 bin yorum içeren etiketli veri seti ile eğitilmiş bir uygulama tasarlanmıştır. Uygulama yapılan yorumları sınıflandırmaktadır.
Bunu yaparken Türkçe olmayan kelimeleri tespit edip bunları Türkçesi ile değiştirmekte, belirli uzunluktan fazla olan yorumları özetlemekte, anlamsız/fake yorumları tespit edebilmekte ve Anlam karmaşası olan yorumlara karşı en başarılı sınıflandırmayı yapmayı hedeflemektedir.

Bu projede, Türkçe olmayan kelimelerin tespiti ve bu kelimelerin uygun Türkçe karşılıkları ile değiştirilmesi sağlanmaktadır. Çeviri sürecinde, özel terimlerin ve özel isimlerin doğru şekilde korunması amacıyla, belirli kelimelerin çevrilmemesi gibi özel hususlar dikkate alınmaktadır. Yorumların belirli bir uzunluğun üzerinde olması durumunda, extractive özetleme yöntemi kullanılarak bu yorumlar özetlenmektedir. Ayrıca, anlamsız veya sahte yorumların tespit edilmesine yönelik veri temizleme ve anomali tespiti yöntemleri uygulanmaktadır. Projenin temel hedefi, anlam karmaşası barındıran yorumlarda en doğru sınıflandırmanın yapılmasını sağlayarak, dilin doğal varyasyonlarına karşı dayanıklı bir model geliştirmektir.

![image](https://github.com/user-attachments/assets/6c2550fc-7064-48a7-aa8a-e47d289f943e)


Türkçenin yapısal olarak zor bir olarak görülmesi, LLM'lerin çoğunun Avrupa dillerinde daha başarılı olması bu çalışmanın önemini artırmaktadır.
Ayrıca Türkçe dilinin korunması bakımından, Türkçe içerisinde sıklıkla kullanılan yabancı kelimelerede değinmekte bunları Türkçe kullanmaya teşvik etmektedir.

Sunulan Projenin kodları uygulama.py'de içerisindedir.
