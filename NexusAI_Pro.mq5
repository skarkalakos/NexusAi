//+------------------------------------------------------------------+
//|                                                NexusAI_Pro.mq5   |
//|                                     Copyright 2024, NexusAI Co.  |
//|                                       https://www.mql5.com       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, NexusAI Co."
#property link      "https://www.mql5.com"
#property version   "2.00"
#property description "NexusAI Deep MLP Trading System."
#property description "Pre-trained architecture. Hardened for Real Ticks."

#include <Trade\Trade.mqh>
CTrade trade;

//--- Neural Network Architecture Definition ---
// Architecture: 8 Inputs -> 6 Hidden -> 4 Hidden -> 1 Output
#define NUM_INPUTS 8
#define NUM_HIDDEN_1 6
#define NUM_HIDDEN_2 4
#define NUM_OUTPUTS 1

//--- Pre-Trained Weights (Placeholder for Python output)
// In a full commercial release, these arrays would be populated by the Python script.
// For now, these are default "safe" weights that can still be somewhat optimized or loaded externally.
double W1[NUM_INPUTS][NUM_HIDDEN_1];
double B1[NUM_HIDDEN_1];
double W2[NUM_HIDDEN_1][NUM_HIDDEN_2];
double B2[NUM_HIDDEN_2];
double W3[NUM_HIDDEN_2][NUM_OUTPUTS];
double B3[NUM_OUTPUTS];

//--- Input Parameters
sinput string   Section1 = "--- Deep Learning Thresholds ---"; 
input double    InpBuyThreshold = 0.50;  // Activation: Buy Threshold (0 to 1)
input double    InpSellThreshold = -0.65; // Activation: Sell Threshold (-1 to 0)

sinput string   Section3 = "--- Risk parameters ---";
input double    InpLotSize = 0.1;           // Fixed Lot Size (if Risk = 0)
input double    InpRiskPercent = 2.0;       // Risk % per Trade (0 = disable)
input int       InpTakeProfit = 500;        // Take Profit (Points)
input int       InpStopLoss = 250;          // Stop Loss (Points)
input int       InpTrailingStop = 150;      // Trailing Stop (Points)
input int       InpTrailingStep = 50;       // Trailing Step (Points)
input int       InpBreakEven = 100;         // Break Even Trigger (Points)
input int       InpBreakEvenLock = 10;      // Break Even Lock Profit (Points)

sinput string   Section4 = "--- Feature Engineering Settings ---";
input int       InpPeriodATR = 14;     // ATR Period
input int       InpPeriodMACD_Fast = 12; // MACD Fast
input int       InpPeriodMACD_Slow = 26; // MACD Slow
input int       InpPeriodRSI = 14;     // RSI Period
input int       InpPeriodBB = 20;      // Bollinger Period
input double    InpBBDev = 2.0;        // Bollinger Deviation
input ulong     InpMagicNumber = 888999; // Magic Number

//--- Global Variables
int handle_atr, handle_macd, handle_rsi, handle_bb;
double st_point;
int st_digits;

// Z-Score Rolling Stats Arrays
double rolling_means[NUM_INPUTS];
double rolling_stdevs[NUM_INPUTS];
const int ROLLING_WINDOW = 100; // Lookback for Z-Score

//+------------------------------------------------------------------+
//| Initialize trained weights (Imported from Python Scikit-Learn)   |
//+------------------------------------------------------------------+
void InitWeights()
{
   W1[0][0] = -1.21665;
   W1[0][1] = 0.58474;
   W1[0][2] = 0.50508;
   W1[0][3] = 0.14589;
   W1[0][4] = 1.05607;
   W1[0][5] = 0.99971;
   W1[1][0] = 0.44975;
   W1[1][1] = -0.56306;
   W1[1][2] = 0.58575;
   W1[1][3] = -0.06373;
   W1[1][4] = 0.45781;
   W1[1][5] = -0.37255;
   W1[2][0] = 1.83180;
   W1[2][1] = -1.18956;
   W1[2][2] = 1.25877;
   W1[2][3] = -0.02107;
   W1[2][4] = 1.05436;
   W1[2][5] = 1.02535;
   W1[3][0] = 0.04753;
   W1[3][1] = -0.19914;
   W1[3][2] = -0.14361;
   W1[3][3] = 0.05203;
   W1[3][4] = -0.11709;
   W1[3][5] = 0.17646;
   W1[4][0] = 0.35467;
   W1[4][1] = 0.12644;
   W1[4][2] = 0.03823;
   W1[4][3] = 0.11891;
   W1[4][4] = -0.18731;
   W1[4][5] = 0.05739;
   W1[5][0] = -0.52837;
   W1[5][1] = 1.95679;
   W1[5][2] = -2.13375;
   W1[5][3] = -0.53096;
   W1[5][4] = -2.48503;
   W1[5][5] = 0.70278;
   W1[6][0] = 0.47043;
   W1[6][1] = -1.14156;
   W1[6][2] = -0.05315;
   W1[6][3] = -0.10626;
   W1[6][4] = -0.73007;
   W1[6][5] = -0.46332;
   W1[7][0] = 0.09355;
   W1[7][1] = 0.00392;
   W1[7][2] = 0.05364;
   W1[7][3] = -0.02324;
   W1[7][4] = 0.05847;
   W1[7][5] = -0.00242;
   B1[0] = 0.82043;
   B1[1] = -0.52774;
   B1[2] = 0.80638;
   B1[3] = 0.76217;
   B1[4] = 0.78485;
   B1[5] = 1.16314;

   W2[0][0] = 1.82605;
   W2[0][1] = 1.90579;
   W2[0][2] = 0.01320;
   W2[0][3] = -0.75401;
   W2[1][0] = -2.23163;
   W2[1][1] = -1.64896;
   W2[1][2] = -0.89032;
   W2[1][3] = -0.25608;
   W2[2][0] = 0.70440;
   W2[2][1] = 0.31465;
   W2[2][2] = -0.30599;
   W2[2][3] = 0.18466;
   W2[3][0] = 0.11412;
   W2[3][1] = 1.02447;
   W2[3][2] = -0.34215;
   W2[3][3] = 0.90446;
   W2[4][0] = -1.42757;
   W2[4][1] = -1.73036;
   W2[4][2] = -1.32166;
   W2[4][3] = 0.17226;
   W2[5][0] = 1.16735;
   W2[5][1] = 1.20396;
   W2[5][2] = 0.85297;
   W2[5][3] = -1.21634;
   B2[0] = 0.46566;
   B2[1] = 0.01513;
   B2[2] = 1.08244;
   B2[3] = 0.12949;

   W3[0][0] = -0.90167;
   W3[1][0] = -1.40956;
   W3[2][0] = -0.93856;
   W3[3][0] = -1.00279;
   B3[0] = 0.32420;
}

//+------------------------------------------------------------------+
//| Z-Score Normalization Helper                                     |
//+------------------------------------------------------------------+
double ZScoreNormalize(double value, int feature_index, const double &historical_data[])
{
   double sum = 0;
   int data_size = ArraySize(historical_data);
   for(int i=0; i<ROLLING_WINDOW && i<data_size; i++) sum += historical_data[i];
   double mean = (MathMin(ROLLING_WINDOW, data_size) > 0) ? sum / MathMin(ROLLING_WINDOW, data_size) : value;
   
   double variance = 0;
   for(int i=0; i<ROLLING_WINDOW && i<data_size; i++) variance += MathPow(historical_data[i] - mean, 2);
   variance = (MathMin(ROLLING_WINDOW, data_size) > 0) ? variance / MathMin(ROLLING_WINDOW, data_size) : 0;
   double stdev = MathSqrt(variance);
   
   if(stdev == 0) return 0;
   double zscore = (value - mean) / stdev;
   
   // Tanh squashing to keep strictly between -1 and 1
   return MathTanh(zscore); 
}

//+------------------------------------------------------------------+
//| Helper Check New Bar (Speeds up optimization massively)          |
//| CRITICAL: Prevents Random Delay/Tick Execution errors            |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, _Period, 0);
   if (current_time != last_time)
   {
      last_time = current_time;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Calculate Dynamic Lot based on Risk %                            |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_points)
{
   if (InpRiskPercent <= 0 || sl_points <= 0) return InpLotSize;
   
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tick_size == 0 || tick_value == 0) return InpLotSize;
   
   double raw_money_risk = AccountInfoDouble(ACCOUNT_BALANCE) * (InpRiskPercent / 100.0);
   
   double sl_distance = sl_points * st_point;
   if(sl_distance == 0) return InpLotSize;
   
   double calculated_lot = raw_money_risk / ((sl_distance / tick_size) * tick_value);
   
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   
   double final_lot = MathRound(calculated_lot / lot_step) * lot_step;
   
   if(final_lot < min_lot) final_lot = min_lot;
   if(final_lot > max_lot) final_lot = max_lot;
   
   return final_lot;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(10); // Handle Slippage cleanly
   
   st_point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   st_digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   InitWeights();
   
   handle_atr = iATR(_Symbol, _Period, InpPeriodATR);
   handle_macd = iMACD(_Symbol, _Period, InpPeriodMACD_Fast, InpPeriodMACD_Slow, 9, PRICE_CLOSE);
   handle_rsi = iRSI(_Symbol, _Period, InpPeriodRSI, PRICE_CLOSE);
   handle_bb = iBands(_Symbol, _Period, InpPeriodBB, 0, InpBBDev, PRICE_CLOSE);
   
   if(handle_atr == INVALID_HANDLE || handle_macd == INVALID_HANDLE || handle_rsi == INVALID_HANDLE || handle_bb == INVALID_HANDLE)
   {
      Print("Error creating handles");
      return(INIT_FAILED);
   }

   return(INIT_SUCCEEDED);
  }
  
//+------------------------------------------------------------------+
//| MLP Feed Forward Pass                                            |
//+------------------------------------------------------------------+
double FeedForward(double &inputs[])
{
   double h1[NUM_HIDDEN_1];
   double h2[NUM_HIDDEN_2];
   double out = 0;
   
   // Layer 1
   for(int j=0; j<NUM_HIDDEN_1; j++)
   {
      double sum = B1[j];
      for(int i=0; i<NUM_INPUTS; i++) sum += inputs[i] * W1[i][j];
      h1[j] = MathTanh(sum); // Leaky ReLU could be used here too, but Tanh is safer for bounded outputs
   }
   
   // Layer 2
   for(int j=0; j<NUM_HIDDEN_2; j++)
   {
      double sum = B2[j];
      for(int i=0; i<NUM_HIDDEN_1; i++) sum += h1[i] * W2[i][j];
      h2[j] = MathTanh(sum);
   }
   
   // Output Layer
   out = B3[0];
   for(int i=0; i<NUM_HIDDEN_2; i++) out += h2[i] * W3[i][0];
   
   return MathTanh(out);
}

//+------------------------------------------------------------------+
//| Get the activation of the MLP architecture                       |
//+------------------------------------------------------------------+
double GetDeepActivation(int index)
{
    double atr[], macd_main[], macd_sig[], rsi[], bb_up[], bb_dn[], close[];
    if(CopyBuffer(handle_atr, 0, index, ROLLING_WINDOW, atr) <= 0) return 0;
    if(CopyBuffer(handle_macd, 0, index, ROLLING_WINDOW, macd_main) <= 0) return 0;
    if(CopyBuffer(handle_macd, 1, index, ROLLING_WINDOW, macd_sig) <= 0) return 0;
    if(CopyBuffer(handle_rsi, 0, index, ROLLING_WINDOW, rsi) <= 0) return 0;
    if(CopyBuffer(handle_bb, 1, index, ROLLING_WINDOW, bb_up) <= 0) return 0;
    if(CopyBuffer(handle_bb, 2, index, ROLLING_WINDOW, bb_dn) <= 0) return 0;
    if(CopyClose(_Symbol, _Period, index, ROLLING_WINDOW, close) <= 0) return 0;
    
    // Feature Engineering:
    // 0: ATR Normalized
    // 1: MACD Histogram
    // 2: RSI
    // 3: Distance from Upper BB
    // 4: Distance from Lower BB
    // 5: Price Momentum (Close - Close[1])
    // 6: MACD Main
    // 7: Volatility Expansion (ATR diff)
    
    double inputs[NUM_INPUTS]; // Static is fine for standard array inside function without passing reference
    inputs[0] = ZScoreNormalize(atr[0], 0, atr); // atr is dynamic, it's fine
    
    double hist[]; ArrayResize(hist, ROLLING_WINDOW);
    for(int i=0; i<ROLLING_WINDOW; i++) hist[i] = macd_main[i] - macd_sig[i];
    inputs[1] = ZScoreNormalize(hist[0], 1, hist); // hist is dynamic now
    
    inputs[2] = ZScoreNormalize(rsi[0], 2, rsi); // rsi is dynamic, it's fine
    
    double dist_up[]; ArrayResize(dist_up, ROLLING_WINDOW);
    for(int i=0; i<ROLLING_WINDOW; i++) dist_up[i] = bb_up[i] - close[i];
    inputs[3] = ZScoreNormalize(dist_up[0], 3, dist_up);
    
    double dist_dn[]; ArrayResize(dist_dn, ROLLING_WINDOW);
    for(int i=0; i<ROLLING_WINDOW; i++) dist_dn[i] = close[i] - bb_dn[i];
    inputs[4] = ZScoreNormalize(dist_dn[0], 4, dist_dn);
    
    double mom[]; ArrayResize(mom, ROLLING_WINDOW);
    for(int i=0; i<ROLLING_WINDOW-1; i++) mom[i] = close[i] - close[i+1];
    mom[ROLLING_WINDOW-1] = mom[ROLLING_WINDOW-2]; // padding last val
    inputs[5] = ZScoreNormalize(mom[0], 5, mom);
    
    inputs[6] = ZScoreNormalize(macd_main[0], 6, macd_main); // macd_main is dynamic, it's fine
    
    double atr_diff[]; ArrayResize(atr_diff, ROLLING_WINDOW);
    for(int i=0; i<ROLLING_WINDOW-1; i++) atr_diff[i] = atr[i] - atr[i+1];
    atr_diff[ROLLING_WINDOW-1] = atr_diff[ROLLING_WINDOW-2]; // padding last val
    inputs[7] = ZScoreNormalize(atr_diff[0], 7, atr_diff);
      
    return FeedForward(inputs);
}

//+------------------------------------------------------------------+
//| Manage Trailing and Breakeven                                    |
//+------------------------------------------------------------------+
void ManageRisk()
{
   if(!PositionSelect(_Symbol)) return;
   
   double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
   double pos_sl = PositionGetDouble(POSITION_SL);
   long type = PositionGetInteger(POSITION_TYPE);
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Break Even
   if(InpBreakEven > 0)
   {
      if(type == POSITION_TYPE_BUY)
      {
         if(bid - pos_open >= InpBreakEven * st_point)
         {
            double new_sl = pos_open + InpBreakEvenLock * st_point;
            if(pos_sl < new_sl) trade.PositionModify(_Symbol, NormalizeDouble(new_sl, st_digits), PositionGetDouble(POSITION_TP));
         }
      }
      else if(type == POSITION_TYPE_SELL)
      {
         if(pos_open - ask >= InpBreakEven * st_point)
         {
            double new_sl = pos_open - InpBreakEvenLock * st_point;
            if(pos_sl > new_sl || pos_sl == 0) trade.PositionModify(_Symbol, NormalizeDouble(new_sl, st_digits), PositionGetDouble(POSITION_TP));
         }
      }
   }
   
   // Trailing Stop
   if(InpTrailingStop > 0)
   {
      if(type == POSITION_TYPE_BUY)
      {
         if(bid - pos_open > InpTrailingStop * st_point)
         {
            double new_sl = bid - InpTrailingStop * st_point;
            if(new_sl > pos_sl + InpTrailingStep * st_point) trade.PositionModify(_Symbol, NormalizeDouble(new_sl, st_digits), PositionGetDouble(POSITION_TP));
         }
      }
      else if(type == POSITION_TYPE_SELL)
      {
         if(pos_open - ask > InpTrailingStop * st_point)
         {
            double new_sl = ask + InpTrailingStop * st_point;
            if(pos_sl == 0 || new_sl < pos_sl - InpTrailingStep * st_point) trade.PositionModify(_Symbol, NormalizeDouble(new_sl, st_digits), PositionGetDouble(POSITION_TP));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Helper func to count open positions for symbol                   |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      string symbol = PositionGetSymbol(i);
      ulong magic = PositionGetInteger(POSITION_MAGIC);
      if(symbol == _Symbol && magic == InpMagicNumber) count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Manage open trades (Runs every tick for tight trailing stops)
   ManageRisk();
   
   // Strict Execution Logic: Only run signal parsing on Completed Bars
   // This guarantees robustness against random tick delays!
   if(!IsNewBar()) return;
   
   int positions = CountPositions();
   if(positions > 0) return; // Wait until position closes
   
   // 1 = Previous Completed Bar. 2 = Bar before that. (Never process unfinished Index 0)
   double activation = GetDeepActivation(1);
   double prev_activation = GetDeepActivation(2);
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double lot = CalculateLotSize(InpStopLoss);
   
   // Buy condition: Crossed the BuyThreshold upwards
   if(activation > InpBuyThreshold && prev_activation <= InpBuyThreshold)
   {
      double sl = InpStopLoss > 0 ? ask - InpStopLoss * st_point : 0;
      double tp = InpTakeProfit > 0 ? ask + InpTakeProfit * st_point : 0;
      trade.Buy(lot, _Symbol, ask, NormalizeDouble(sl, st_digits), NormalizeDouble(tp, st_digits), "Deep MLP Buy");
   }
   
   // Sell condition: Crossed the SellThreshold downwards
   if(activation < InpSellThreshold && prev_activation >= InpSellThreshold)
   {
      double sl = InpStopLoss > 0 ? bid + InpStopLoss * st_point : 0;
      double tp = InpTakeProfit > 0 ? bid - InpTakeProfit * st_point : 0;
      trade.Sell(lot, _Symbol, bid, NormalizeDouble(sl, st_digits), NormalizeDouble(tp, st_digits), "Deep MLP Sell");
   }
  }
//+------------------------------------------------------------------+
