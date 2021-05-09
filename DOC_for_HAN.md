make[2]: Leaving directory '/home/all/Han/mecab-0.996-ko-0.9.2/tests'
make[1]: Leaving directory '/home/all/Han/mecab-0.996-ko-0.9.2/tests'
make[1]: Entering directory '/home/all/Han/mecab-0.996-ko-0.9.2'
make[1]: Leaving directory '/home/all/Han/mecab-0.996-ko-0.9.2'


echo To enable di
ary, rewrite /usr/local/etc/mecabrc as \"dicdir = /usr/local/lib/mecab/dic/mecab-ko-dic\"
To enable dictionary, rewrite /usr/local/etc/mecabrc as "dicdir = /usr/local/lib/mecab/dic/mecab-ko-dic"
/usr/local/lib/mecab/dic/mecab-ipadic-neologd

https://dumps.wikimedia.org/kowiki/latest/


ssh all@192.168.1.129






window=5, iteration=10, negative=15


--dim-size 300 --window 5 --iteration 10 --negative 15 --lowercase False --tokenizer mecab

--tokenizer: The name of the tokenizer used to tokenize a text into words. Possible choices are regexp, icu, mecab, and jieba
   mecab
--sent-detect: The sentence detector used to split texts into sentences. Currently, only icu is the possible value (default: None)
--min-word-count: A word is ignored if the total frequency of the word is less than this value (default: 10)
--min-entity-count: An entity is ignored if the total frequency of the entity appearing as the referent of an anchor link is less than this value (default: 5)
--min-paragraph-len: A paragraph is ignored if its length is shorter than this value (default: 5)
--category/--no-category: Whether to include Wikipedia categories in the dictionary (default:False)
--disambi/--no-disambi: Whether to include disambiguation entities in the dictionary (default:False)
--link-graph/--no-link-graph: Whether to learn from the Wikipedia link graph (default: True)
--entities-per-page: For processing each page, the specified number of randomly chosen entities are used to predict their neighboring entities in the link graph (default: 10)
--link-mentions: Whether to convert entity names into links (default: True)
--min-link-prob: An entity name is ignored if the probability of the name appearing as a link is less than this value (default: 0.2)
--min-prior-prob: An entity is not registered as a referent of an entity name if the probability of the entity name referring to the entity is less than this value (default: 0.01)
--max-mention-len: The maximum number of characters in an entity name (default: 20)
--init-alpha: The initial learning rate (default: 0.025)
--min-alpha: The minimum learning rate (default: 0.0001)
--sample: The parameter that controls the downsampling of frequent words (default: 1e-4)
--word-neg-power: Negative sampling of words is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0.75)
--entity-neg-power: Negative sampling of entities is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0)
--pool-size: The number of worker processes (default: the number of CPUs)