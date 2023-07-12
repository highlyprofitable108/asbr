graph LR
  subgraph "Common Python Scripts"
    A1[api_utils]
    A2[model_utils]
    A3[dataframe_utils]
    A4[database_utils]
    A5[file_utils]
  end
  subgraph "Sport: Golf"
    B1[golf_db]
    B2[golf_constants]
    B3[golf_features]
    B4[golf_populate_data]
    B5[golf_EDA]
    B6[golf_model]
    B7[golf_review]
    B8[golf_sim_runner]
    B9[golf_sim_results]
    B1 --> B4
    B4 -->|Updates| B1
    B4 --> B5
    B5 --> B6
    B6 --> B7
    B7 --> B8
    B8 --> B9
    B2 -->|Provides vars & paths| B4
    B3 -->|Provides feature cols| B4
    B2 -->|Provides vars & paths| B5
    B3 -->|Provides feature cols| B5
    B2 -->|Provides vars & paths| B6
    B3 -->|Provides feature cols| B6
    B2 -->|Provides vars & paths| B7
    B3 -->|Provides feature cols| B7
    B2 -->|Provides vars & paths| B8
    B3 -->|Provides feature cols| B8
    B2 -->|Provides vars & paths| B9
    B3 -->|Provides feature cols| B9
  end
  subgraph "Sport: NFL"
    C1[nfl_db]
    C2[nfl_constants]
    C3[nfl_features]
    C4[nfl_populate_data]
    C5[nfl_EDA]
    C6[nfl_model]
    C7[nfl_review]
    C8[nfl_sim_runner]
    C9[nfl_sim_results]
    C1 --> C4
    C4 -->|Updates| C1
    C4 --> C5
    C5 --> C6
    C6 --> C7
    C7 --> C8
    C8 --> C9
    C2 -->|Provides vars & paths| C4
    C3 -->|Provides feature cols| C4
    C2 -->|Provides vars & paths| C5
    C3 -->|Provides feature cols| C5
    C2 -->|Provides vars & paths| C6
    C3 -->|Provides feature cols| C6
    C2 -->|Provides vars & paths| C7
    C3 -->|Provides feature cols| C7
    C2 -->|Provides vars & paths| C8
    C3 -->|Provides feature cols| C8
    C2 -->|Provides vars & paths| C9
    C3 -->|Provides feature cols| C9
  end
  subgraph "Sport: College Football"
    D1[cfb_db]
    D2[cfb_constants]
    D3[cfb_features]
    D4[cfb_populate_data]
    D5[cfb_EDA]
    D6[cfb_model]
    D7[cfb_review]
    D8[cfb_sim_runner]
    D9[cfb_sim_results]
    D1 --> D4
    D4 -->|Updates| D1
    D4 --> D5
    D5 --> D6
    D6 --> D7
    D7 --> D8
    D8 --> D9
    D2 -->|Provides vars & paths| D4
    D3 -->|Provides feature cols| D4
    D2 -->|Provides vars & paths| D5
    D3 -->|Provides feature cols| D5
    D2 -->|Provides vars & paths| D6
    D3 -->|Provides feature cols| D6
    D2 -->|Provides vars & paths| D7
    D3 -->|Provides feature cols| D7
    D2 -->|Provides vars & paths| D8
    D3 -->|Provides feature cols| D8
    D2 -->|Provides vars & paths| D9
    D3 -->|Provides feature cols| D9
  end
  subgraph "Sport: College Basketball"
    E1[cbb_db]
    E2[cbb_constants]
    E3[cbb_features]
    E4[cbb_populate_data]
    E5[cbb_EDA]
    E6[cbb_model]
    E7[cbb_review]
    E8[cbb_sim_runner]
    E9[cbb_sim_results]
    E1 --> E4
    E4 -->|Updates| E1
    E4 --> E5
    E5 --> E6
    E6 --> E7
    E7 --> E8
    E8 --> E9
    E2 -->|Provides vars & paths| E4
    E3 -->|Provides feature cols| E4
    E2 -->|Provides vars & paths| E5
    E3 -->|Provides feature cols| E5
    E2 -->|Provides vars & paths| E6
    E3 -->|Provides feature cols| E6
    E2 -->|Provides vars & paths| E7
    E3 -->|Provides feature cols| E7
    E2 -->|Provides vars & paths| E8
    E3 -->|Provides feature cols| E8
    E2 -->|Provides vars & paths| E9
    E3 -->|Provides feature cols| E9
  end
  subgraph "Sport: Fantasy Football"
    F1[ff_db]
    F2[ff_constants]
    F3[ff_features]
    F4[ff_populate_data]
    F5[ff_EDA]
    F6[ff_model]
    F7[ff_review]
    F8[ff_sim_runner]
    F9[ff_sim_results]
    F1 --> F4
    F4 -->|Updates| F1
    F4 --> F5
    F5 --> F6
    F6 --> F7
    F7 --> F8
    F8 --> F9
    F2 -->|Provides vars & paths| F4
    F3 -->|Provides feature cols| F4
    F2 -->|Provides vars & paths| F5
    F3 -->|Provides feature cols| F5
    F2 -->|Provides vars & paths| F6
    F3 -->|Provides feature cols| F6
    F2 -->|Provides vars & paths| F7
    F3 -->|Provides feature cols| F7
    F2 -->|Provides vars & paths| F8
    F3 -->|Provides feature cols| F8
    F2 -->|Provides vars & paths| F9
    F3 -->|Provides feature cols| F9
  end
B4 --> A1
B5 --> A3
B6 --> A2
B7 --> A5
B8 --> A1
B9 --> A4
C4 --> A1
C5 --> A3
C6 --> A2
C7 --> A5
C8 --> A1
C9 --> A4
D4 --> A1
D5 --> A3
D6 --> A2
D7 --> A5
D8 --> A1
D9 --> A4
E4 --> A1
E5 --> A3
E6 --> A2
E7 --> A5
E8 --> A1
E9 --> A4
F4 --> A1
F5 --> A3
F6 --> A2
F7 --> A5
F8 --> A1
F9 --> A4
