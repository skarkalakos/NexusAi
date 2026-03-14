//+------------------------------------------------------------------+
//|                                                NexusAI_Pro.mq5   |
//|                                     Copyright 2024, NexusAI Co.  |
//|                                       https://www.mql5.com       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, NexusAI Co."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "NexusAI Deep Perceptron Trading System."
#property description "Fully optimized for MT5 Strategy Tester Genetic Algorithm."

#include <Trade\Trade.mqh>
CTrade trade;

//--- Input Parameters
sinput string   Section1 = "--- Neural Network Weights (Optimize These) ---"; 
input double    InpWeight1 = -0.8;   // Opt: Weight 1 (RSI) [-1 to 1]
input double    InpWeight2 = 0.4;    // Opt: Weight 2 (WPR) [-1 to 1]
input double    InpWeight3 = -0.6;   // Opt: Weight 3 (DeMarker) [-1 to 1]
input double    InpWeight4 = -0.3;   // Opt: Weight 4 (Stochastic) [-1 to 1]

sinput string   Section2 = "--- Neural Network Thresholds ---"; 
input double    InpBuyThreshold = 0.50;  // Opt: Buy Threshold (0 to 1)
input double    InpSellThreshold = -0.65; // Opt: Sell Threshold (-1 to 0)

sinput string   Section3 = "--- Risk parameters ---";
input double    InpLotSize = 0.1;           // Fixed Lot Size (if Risk = 0)
input double    InpRiskPercent = 2.0;       // Risk % per Trade (0 = disable)
input int       InpTakeProfit = 500;        // Take Profit (Points)
input int       InpStopLoss = 250;          // Stop Loss (Points)
input int       InpTrailingStop = 150;      // Trailing Stop (Points, 0 = disable)
input int       InpTrailingStep = 50;       // Trailing Step (Points)
input int       InpBreakEven = 100;         // Break Even Trigger (Points, 0 = disable)
input int       InpBreakEvenLock = 10;      // Break Even Lock Profit (Points)

sinput string   Section4 = "--- Optimizer & Filter Settings ---";
input int       InpPeriodRSI = 14;     // RSI Period
input int       InpPeriodWPR = 14;     // WPR Period
input int       InpPeriodDeM = 14;     // DeMarker Period
input int       InpPeriodStoch = 14;   // Stochastic Period K
input ulong     InpMagicNumber = 888999; // Magic Number

//--- Global Variables
int handle_rsi, handle_wpr, handle_dem, handle_stoch;
double st_point;
int st_digits;

//+------------------------------------------------------------------+
//| Helper Check New Bar (Speeds up optimization massively)          |
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
   
   // Sl Points means points * _Point. We need price difference:
   double sl_distance = sl_points * st_point;
   if(sl_distance == 0) return InpLotSize;
   
   // Formula: Lot = RiskAmount / (SL_Distance / TickSize * TickValue)
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
   
   st_point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   st_digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   // Handle Initialization
   handle_rsi = iRSI(_Symbol, _Period, InpPeriodRSI, PRICE_CLOSE);
   handle_wpr = iWPR(_Symbol, _Period, InpPeriodWPR);
   handle_dem = iDeMarker(_Symbol, _Period, InpPeriodDeM);
   handle_stoch = iStochastic(_Symbol, _Period, InpPeriodStoch, 3, 3, MODE_SMA, STO_LOWHIGH);
   
   if(handle_rsi == INVALID_HANDLE || handle_wpr == INVALID_HANDLE || handle_dem == INVALID_HANDLE || handle_stoch == INVALID_HANDLE)
   {
      Print("Error creating oscillator handles");
      return(INIT_FAILED);
   }

   return(INIT_SUCCEEDED);
  }
  
//+------------------------------------------------------------------+
//| Get the activation of the Perceptron for a specific index        |
//+------------------------------------------------------------------+
double GetPerceptronActivation(int index)
{
    double rsi[], wpr[], dem[], stoch[];
    if(CopyBuffer(handle_rsi, 0, index, 1, rsi) <= 0) return 0;
    if(CopyBuffer(handle_wpr, 0, index, 1, wpr) <= 0) return 0;
    if(CopyBuffer(handle_dem, 0, index, 1, dem) <= 0) return 0;
    if(CopyBuffer(handle_stoch, 0, index, 1, stoch) <= 0) return 0;
    
    double norm_rsi = (rsi[0] - 50.0) / 50.0;
    double norm_wpr = (wpr[0] + 50.0) / 50.0; // WPR is -100 to 0
    double norm_dem = (dem[0] - 0.5) / 0.5;
    double norm_stoch = (stoch[0] - 50.0) / 50.0;
      
    double sum = (InpWeight1 * norm_rsi) + 
                 (InpWeight2 * norm_wpr) + 
                 (InpWeight3 * norm_dem) + 
                 (InpWeight4 * norm_stoch);
                 
    return MathTanh(sum);
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
   
   // Only scan for signals on new bar to save computing power
   if(!IsNewBar()) return;
   
   int positions = CountPositions();
   if(positions > 0) return; // Wait until position closes
   
   // Index 1 = Previously closed bar to prevent repainting in calculation
   double activation = GetPerceptronActivation(1);
   double prev_activation = GetPerceptronActivation(2);
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double lot = CalculateLotSize(InpStopLoss);
   
   // Buy condition: Crossed the BuyThreshold upwards
   if(activation > InpBuyThreshold && prev_activation <= InpBuyThreshold)
   {
      double sl = InpStopLoss > 0 ? ask - InpStopLoss * st_point : 0;
      double tp = InpTakeProfit > 0 ? ask + InpTakeProfit * st_point : 0;
      trade.Buy(lot, _Symbol, ask, NormalizeDouble(sl, st_digits), NormalizeDouble(tp, st_digits), "NexusAI Buy Signal");
   }
   
   // Sell condition: Crossed the SellThreshold downwards
   if(activation < InpSellThreshold && prev_activation >= InpSellThreshold)
   {
      double sl = InpStopLoss > 0 ? bid + InpStopLoss * st_point : 0;
      double tp = InpTakeProfit > 0 ? bid - InpTakeProfit * st_point : 0;
      trade.Sell(lot, _Symbol, bid, NormalizeDouble(sl, st_digits), NormalizeDouble(tp, st_digits), "NexusAI Sell Signal");
   }
  }
//+------------------------------------------------------------------+
