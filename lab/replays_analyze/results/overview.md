# Replay Analysis — Overview

## Store: 6269 episodes, 4,513,680 total steps
- `steps`: 4,913,680 rows
- `farm_steps`: 9,827,360 rows
- `private_steps`: 9,827,360 rows
- `actions`: 97,218,335 rows
- `market_orders`: 11,913,755 rows
- `tiles_delta`: 59,460,704 rows

## Score distribution (all episodes, both players)
- games: 12,538 player-scores; min=24094 p25=69163 median=86790 p75=104734 p90=123030 p99=148628 max=178015

### Top agents (min 3 games)
| agent            |   games |   avg_money |   best |
|:-----------------|--------:|------------:|-------:|
| Less             |       3 |      105676 | 130530 |
| shinamonmon      |       7 |      103648 | 146171 |
| Lenin Goud       |       8 |      102774 | 138771 |
| Raef Guizani     |       4 |      101298 | 145313 |
| jasonstillchasin |      18 |      100618 | 136486 |
| Epiphany_Thu     |       4 |       98893 | 133333 |
| Junichiro Morita |      74 |       98258 | 159110 |
| Crop Dusta       |     351 |       98137 | 165021 |
| VanKoha          |      27 |       98060 | 138822 |
| Aaweg            |      26 |       96879 | 136954 |
| lllleeeo         |      18 |       96769 | 127518 |
| 22307110257-张扬   |       7 |       96345 | 122752 |
| MiMi             |     105 |       96112 | 159846 |
| ActiveMusyoku    |      59 |       95719 | 139078 |
| muyouqian4       |      19 |       94098 | 146284 |

## Crop mix: winners vs losers (PLANT actions)
| crop       | side   |   plants |
|:-----------|:-------|---------:|
| CARROT     | loser  |    42930 |
| CARROT     | winner |    52683 |
| MELON      | loser  |    98824 |
| MELON      | winner |    90964 |
| STRAWBERRY | loser  |   236891 |
| STRAWBERRY | winner |   231813 |
| TOMATO     | loser  |     1022 |
| TOMATO     | winner |     3917 |
| WHEAT      | loser  |   811610 |
| WHEAT      | winner |   792049 |

## Sell timing by product (winners, day distribution p25/p50/p75)
| item       |     p25 |     p50 |     p75 |   total_sells |
|:-----------|--------:|--------:|--------:|--------------:|
| FERTILIZER | 10.7604 | 17.1458 | 23.5312 |        903360 |
| WHEAT      | 12.0208 | 18      | 23.9792 |        779386 |
| MILK       | 13.6771 | 19.1042 | 24.5312 |        246086 |
| STRAWBERRY | 16.6771 | 21.1042 | 25.5312 |        236877 |
| WOOL       | 13.7708 | 19.1667 | 24.5625 |        205625 |
| MELON      | 16.5    | 20.375  | 24.7083 |         64858 |
| CARROT     | 15.4792 | 19.9583 | 27.2292 |         29849 |
| EGG        | 14.7812 | 19.6458 | 24.9688 |          7787 |
| TOMATO     | 20.9792 | 24.6667 | 27.5208 |          4446 |