from cleanRL_implementation import AlgoPlayer
import asyncio

state_path= "./models/testing/fixed_teams"
username="daisy_hearts1"

myPlayer = AlgoPlayer(state_path=state_path,informat="gen9doublesou",inteam="""Charizard||LifeOrb|Blaze|hurricane,heatwave,scorchingsands,protect||85,,85,85,85,85|M|,0,,,,||82|,,,,,Fire]Malamar||SitrusBerry|Contrary|protect,trickroom,knockoff,superpower||85,85,85,85,85,|F|,,,,,0||80|,,,,,Fighting]Hydrapple||ChoiceSpecs|Regenerator|earthpower,leafstorm,gigadrain,dracometeor||85,,85,85,85,85|F|,0,,,,||85|,,,,,Fire]Tornadus||SitrusBerry|Prankster|knockoff,bleakwindstorm,tailwind,heatwave||85,85,85,85,85,85|M|||77|,,,,,Steel]Barraskewda||LifeOrb|PropellerTail|psychicfangs,waterfall,protect,closecombat||85,85,85,85,85,85|F|||85|,,,,,Fighting]Arcanine||HeavyDutyBoots|Intimidate|willowisp,flareblitz,closecombat,morningsun||85,85,85,85,85,85|M|||82|,,,,,Fighting""",filename="agent.pt")
asyncio.run(myPlayer.send_challenges(username,n_challenges=1))