for ($i=0; $i -lt 7; $i++){
    python main.py experiments --iterations 5 --experiments 3 --color-mode gray --out out/gray --blur $i
}