//+------------------------------------------------------------------+
//|                                             NexusAI_Signal.mq5   |
//|                                     Copyright 2024, NexusAI Co.  |
//|                                       https://www.mql5.com       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, NexusAI Co."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "NexusAI Deep Perceptron Logic Visualizer"
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

//--- plot BuySignal
#property indicator_label1  "Buy Signal"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- plot SellSignal
#property indicator_label2  "Sell Signal"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  2

//--- Input Parameters for the Neural Network
sinput string   Section1 = "--- Neural Network Weights ---"; // Network Weights (Genetic Optimization)
input double    InpWeight1 = -0.8;   // Weight 1 (RSI)
input double    InpWeight2 = 0.4;    // Weight 2 (WPR)
input double    InpWeight3 = -0.6;   // Weight 3 (DeMarker)
input double    InpWeight4 = -0.3;   // Weight 4 (Stochastic)

sinput string   Section2 = "--- Neural Network Thresholds ---"; 
input double    InpBuyThreshold = 0.50;   // Buy Activation Threshold (0 to 1)
input double    InpSellThreshold = -0.65; // Sell Activation Threshold (-1 to 0)

sinput string   Section3 = "--- Oscillator Periods ---";
input int       InpPeriodRSI = 14;   // RSI Period
input int       InpPeriodWPR = 14;   // WPR Period
input int       InpPeriodDeM = 14;   // DeMarker Period
input int       InpPeriodStoch = 14; // Stochastic Period K

//--- Indicator Buffers
double         BuySignalBuffer[];
double         SellSignalBuffer[];

//--- Handles for indicators
int            handle_rsi;
int            handle_wpr;
int            handle_dem;
int            handle_stoch;

//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- indicator buffers mapping
   SetIndexBuffer(0, BuySignalBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, SellSignalBuffer, INDICATOR_DATA);
   
//--- set arrow codes
   PlotIndexSetInteger(0, PLOT_ARROW, 233); // Up arrow
   PlotIndexSetInteger(1, PLOT_ARROW, 234); // Down arrow

//--- initialize handles
   handle_rsi = iRSI(_Symbol, _Period, InpPeriodRSI, PRICE_CLOSE);
   if(handle_rsi == INVALID_HANDLE) { Print("Failed to create RSI handle"); return(INIT_FAILED); }
   
   handle_wpr = iWPR(_Symbol, _Period, InpPeriodWPR);
   if(handle_wpr == INVALID_HANDLE) { Print("Failed to create WPR handle"); return(INIT_FAILED); }
   
   handle_dem = iDeMarker(_Symbol, _Period, InpPeriodDeM);
   if(handle_dem == INVALID_HANDLE) { Print("Failed to create DeM handle"); return(INIT_FAILED); }
   
   handle_stoch = iStochastic(_Symbol, _Period, InpPeriodStoch, 3, 3, MODE_SMA, STO_LOWHIGH);
   if(handle_stoch == INVALID_HANDLE) { Print("Failed to create Stoch handle"); return(INIT_FAILED); }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
//--- Ensure we have enough data
   if(rates_total < 50) return(0);
   
   int limit = prev_calculated == 0 ? rates_total - 1 : rates_total - prev_calculated;

//--- Arrays for indicator data
   double rsi[], wpr[], dem[], stoch[];
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(wpr, true);
   ArraySetAsSeries(dem, true);
   ArraySetAsSeries(stoch, true);
   
   ArraySetAsSeries(BuySignalBuffer, true);
   ArraySetAsSeries(SellSignalBuffer, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(high, true);

//--- Copy indicator data
   if(CopyBuffer(handle_rsi, 0, 0, limit + 1, rsi) <= 0) return(0);
   if(CopyBuffer(handle_wpr, 0, 0, limit + 1, wpr) <= 0) return(0);
   if(CopyBuffer(handle_dem, 0, 0, limit + 1, dem) <= 0) return(0);
   if(CopyBuffer(handle_stoch, 0, 0, limit + 1, stoch) <= 0) return(0);

//--- Main loop
   for(int i = limit; i >= 0 && !IsStopped(); i--)
     {
      // Default to empty value
      BuySignalBuffer[i] = EMPTY_VALUE;
      SellSignalBuffer[i] = EMPTY_VALUE;
      
      // Calculate normalized inputs [-1, 1]
      double norm_rsi = (rsi[i] - 50.0) / 50.0;
      double norm_wpr = (wpr[i] + 50.0) / 50.0; // WPR is -100 to 0
      double norm_dem = (dem[i] - 0.5) / 0.5;
      double norm_stoch = (stoch[i] - 50.0) / 50.0;
      
      // Calculate Perceptron Output
      double sum = (InpWeight1 * norm_rsi) + 
                   (InpWeight2 * norm_wpr) + 
                   (InpWeight3 * norm_dem) + 
                   (InpWeight4 * norm_stoch);
                   
      double activation = MathTanh(sum); // Output range: -1 to 1
      
      // Avoid repainting by checking the condition on a completed bar if we want, 
      // but for pure signal viewing, real-time activation is fine.
      
      // Signal logic: Check crossover of threshold
      // To strictly generate signals and not continuous output, we check previous state
      if (i < rates_total - 1) 
      {
         double prev_norm_rsi = (rsi[i+1] - 50.0) / 50.0;
         double prev_norm_wpr = (wpr[i+1] + 50.0) / 50.0;
         double prev_norm_dem = (dem[i+1] - 0.5) / 0.5;
         double prev_norm_stoch = (stoch[i+1] - 50.0) / 50.0;
         double prev_sum = (InpWeight1 * prev_norm_rsi) + (InpWeight2 * prev_norm_wpr) + (InpWeight3 * prev_norm_dem) + (InpWeight4 * prev_norm_stoch);
         double prev_activation = MathTanh(prev_sum);
         
         if (activation > InpBuyThreshold && prev_activation <= InpBuyThreshold)
         {
            BuySignalBuffer[i] = low[i] - 10 * _Point;
         }
         else if (activation < InpSellThreshold && prev_activation >= InpSellThreshold)
         {
            SellSignalBuffer[i] = high[i] + 10 * _Point;
         }
      }
     }
     
   return(rates_total);
  }
//+------------------------------------------------------------------+
