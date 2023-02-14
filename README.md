# Automatický přiřazování 2. signatury ve studovně, na základě názvu a klíčových slov

Klíčová slova a dobré názvy jsou v MARCXML souboru, který jde vygenerovat z Almy. 
Je na to několik metod, buď to jde na základě vyhledávání lokace a druhu dokumentu, 
nebo si vytvoříme XLSX soubor se systémovýma číslama dokumentů, který chceme zpracovat.
To je jistější, protože jinak se množiny dokumentů nemusí shodovat.

XLSX soubor `mms_id.xlsx` můžeme vytvořit z XML souboru staženýho z Alma analytics (třeba pro všechny jednotky ve studovně ve "Statistikách pro správu fondu")  pomocí:

```
$ python sysnostoxlsx.py studovna.xml
```

Pak musíme vytvořit <a href="https://knowledge.exlibrisgroup.com/Alma/Product_Documentation/010Alma_Online_Help_(English)/050Administration/080Managing_Search_Queries_and_Sets">
sadu v Almě</a>. V menu Správce vybereme *Spravovat sady*, tam tlačítkem *Přidat sadu* přidáme *Specifikováno*. 
Ve formuláři vybereme *Fyzické tituly*, nastavíme nějakej název, a nahrajeme soubor *mms_id.xlsx*.

Výslednej soubor je pak ve *Správce* -> *Spravovat exporty*.

Pak můžeme klasifikovat za využití MARCXML souboru a původního XML z analytik:

```
$ python classify.py /home/mint/Stažené/BIBLIOGRAPHIC_16215255210006986_1.xml studovna.xml
```
