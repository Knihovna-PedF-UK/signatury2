# Automatický přiřazování 2. signatury ve studovně, na základě názvu a klíčových slov

Klíčová slova a dobré názvy jsou v MARCXML souboru, který jde vygenerovat z Almy. 
Je na to několik metod, buď to jde na základě vyhledávání lokace a druhu dokumentu, 
nebo si vytvoříme XLSX soubor se systémovýma číslama dokumentů, který chceme zpracovat.
To je jistější, protože jinak se množiny dokumentů nemusí shodovat.



```
$ python classify.py /home/mint/Stažené/BIBLIOGRAPHIC_16215255210006986_1.xml studovna.xml
```
