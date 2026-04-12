import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse

class TimewaveZero:
    """
    Python implementation of Terence McKenna's Timewave Zero algorithm.
    Includes both the original 'Kelley' set (with the half-twist error) 
    and the corrected 'Watkins' set.
    """

    # The 'First Order of Difference' (FOD) from the King Wen Sequence (64 hexagrams)
    # These represent the number of lines that change between successive hexagrams.
    FOD = np.array([
        3, 0, 3, 4, 3, 3, 3, 0, 3, 3, 0, 3, 4, 3, 4, 3,
        3, 0, 3, 3, 4, 3, 0, 3, 4, 4, 0, 6, 2, 2, 6, 4,
        4, 6, 2, 2, 6, 0, 4, 4, 3, 0, 3, 4, 3, 3, 0, 3,
        3, 4, 3, 4, 3, 0, 3, 3, 0, 3, 3, 3, 4, 3, 0, 3
    ])

    def __init__(self, mode='watkins'):
        self.mode = mode.lower()
        self.data = self._generate_384_set()

    def _generate_384_set(self):
        """
        Expands the 64 FOD values into the 384-point data set using McKenna's triplication logic.
        This is where the 'Half Twist' error occurs in the Kelley version.
        """
        # WATKINS SET (Corrected - FULL 384 POINTS)
        watkins = np.array([
            0, 0, 0, 2, 7, 4, 3, 2, 6, 8, 13, 5, 26, 25, 24, 15, 13, 16, 14, 19, 17, 24, 20, 25, 63, 60, 56, 55, 47, 53, 36, 38, 39, 43, 39, 35, 22, 24, 22, 21, 29, 30, 27, 26, 26, 21, 23, 19, 57, 62, 61, 60, 52, 58, 41, 43, 44, 48, 44, 40, 27, 29, 27, 26, 34, 35, 32, 31, 31, 26, 28, 24, 62, 67, 66, 65, 57, 63, 46, 48, 49, 53, 49, 45, 32, 34, 32, 31, 39, 40, 37, 36, 36, 31, 33, 29, 67, 72, 71, 70, 62, 68, 51, 53, 54, 58, 54, 50, 37, 39, 37, 36, 44, 45, 42, 41, 41, 36, 38, 34, 72, 77, 76, 75, 67, 73, 56, 58, 59, 63, 59, 55, 42, 44, 42, 41, 49, 50, 47, 46, 46, 41, 43, 39, 77, 82, 81, 80, 72, 78, 61, 63, 64, 68, 64, 60, 47, 49, 47, 46, 54, 55, 52, 51, 51, 46, 48, 44, 79, 79, 79, 77, 72, 75, 76, 77, 73, 71, 66, 74, 53, 54, 55, 64, 66, 63, 65, 60, 62, 55, 59, 54, 16, 19, 23, 24, 32, 26, 43, 41, 40, 36, 40, 44, 57, 55, 57, 58, 50, 49, 52, 53, 45, 44, 47, 50, 22, 17, 18, 19, 27, 21, 38, 36, 35, 31, 35, 39, 52, 50, 52, 53, 45, 44, 47, 48, 40, 39, 42, 45, 17, 12, 13, 14, 22, 16, 33, 31, 30, 26, 30, 34, 47, 45, 47, 48, 40, 39, 42, 43, 35, 34, 37, 40, 12, 7, 8, 9, 17, 11, 28, 26, 25, 21, 25, 29, 42, 40, 42, 43, 35, 34, 37, 38, 30, 29, 32, 35, 7, 2, 3, 4, 12, 6, 23, 21, 20, 16, 20, 24, 37, 35, 37, 38, 30, 29, 32, 33, 25, 24, 27, 30, 2, 0, 0, 0, 7, 1, 18, 16, 15, 11, 15, 19, 32, 30, 32, 33, 25, 24, 27, 28, 20, 19, 22, 25, 0, 0, 0, 2, 7, 4, 3, 2, 6, 8, 13, 5, 26, 25, 24, 15, 13, 16, 14, 19, 17, 24, 20, 25, 63, 60, 56, 55, 47, 53, 36, 38, 39, 43, 39, 35, 22, 24, 22, 21, 29, 30, 27, 26, 26, 21, 23, 19, 57, 62, 61, 60, 52, 58, 41, 43, 44, 48, 44, 40, 27, 29, 27, 26, 34, 35, 32, 31, 31, 26, 28, 24, 62, 67, 66, 65, 57, 63, 46, 48, 49, 53, 49, 45, 32, 34, 32, 31, 39, 40, 37, 36, 36, 31, 33, 29, 67, 72, 71, 70, 62, 68, 51, 53, 54, 58, 54, 50, 37, 39, 37, 36, 44, 45, 42, 41, 41, 36, 38, 34, 72, 77, 76, 75, 67, 73, 56, 58, 59, 63, 59, 55, 42, 44, 42, 41, 49, 50, 47, 46, 46, 41, 43, 39, 77, 82, 81, 80, 72, 78, 61, 63, 64, 68, 64, 60, 47, 49, 47, 46, 54, 55, 52, 51, 51, 46, 48, 44
        ])

        if self.mode == 'watkins':
            return watkins
        else:
            kelley = np.copy(watkins)
            first_half = watkins[:192]
            second_half = first_half[::-1] 
            kelley[192:] = second_half
            return kelley

    def get_novelty(self, days_to_zero):
        """
        Calculates the novelty value for a given number of days before the zero date.
        The algorithm is a fractal summation of the data set across powers of 64.
        """
        # Novelty value W(t) = Sum [ v(t * 64^-k % 384) / 64^k ]
        total_novelty = 0.0
        # McKenna's software typically summed 13 or more levels.
        # k=0 is the 'day' level, k=1 is the '64-day' level, etc.
        for k in range(-3, 13): 
            divisor = 64.0**k
            index = (days_to_zero / divisor) % 384
            
            idx_low = int(index)
            idx_high = (idx_low + 1) % 384
            frac = index - idx_low
            
            val = (1 - frac) * self.data[idx_low] + frac * self.data[idx_high]
            total_novelty += val / divisor
            
        return total_novelty

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

def generate_graph(zero_date, start_date, mode, filename="novelty_chart.png"):
    tw = TimewaveZero(mode=mode)
    
    total_days = (zero_date - start_date).days
    x_dates = []
    y_vals = []
    
    # Generate ~500 points for a smooth curve
    steps = 500
    for i in range(steps + 1):
        current_date = start_date + timedelta(days=(total_days * i / steps))
        days_to_zero = (zero_date - current_date).total_seconds() / (24 * 3600)
        
        # Don't go past zero as it gets messy
        if days_to_zero < 0: days_to_zero = 0
            
        novelty = tw.get_novelty(days_to_zero)
        x_dates.append(current_date)
        y_vals.append(novelty)
        
    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, y_vals, color='cyan', linewidth=1.5)
    
    # McKenna style: Lower Y is MORE novelty, higher Y is MORE habit.
    # We'll invert the Y axis to show the wave 'plunging' into novelty.
    plt.gca().invert_yaxis()
    
    plt.title(f"Timewave Zero ({mode.upper()}) - Novelty Theory", fontsize=14, color='white')
    plt.xlabel("Historical Time", fontsize=12, color='white')
    plt.ylabel("Degree of Novelty (Lower is More Novel)", fontsize=12, color='white')
    
    # Aesthetics for a 'high-tech' feel
    plt.gcf().set_facecolor('#111111')
    plt.gca().set_facecolor('#222222')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tick_params(colors='white')
    
    plt.tight_layout()
    plt.savefig(filename, facecolor='#111111')
    print(f"\nGraph saved as: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timewave Zero / Novelty Theory Calculator")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--zero", type=str, default="2012-12-21", help="The 'Zero Date' (default: 2012-12-21)")
    parser.add_argument("--mode", choices=['watkins', 'kelley'], default='watkins', help="Algorithm version (watkins=fixed, kelley=original error)")
    parser.add_argument("--plot", action="store_true", help="Generate a graph leading up to the zero date.")
    parser.add_argument("--years", type=int, default=100, help="Number of years to plot before the zero date (default: 100)")
    
    args = parser.parse_args()
    
    zero_date = parse_date(args.zero)
    target_date = parse_date(args.date) if args.date else datetime.now()
    
    # Calculate days remaining (or passed) relative to zero date
    delta = zero_date - target_date
    days_to_zero = delta.total_seconds() / (24 * 3600)
    
    tw = TimewaveZero(mode=args.mode)
    novelty = tw.get_novelty(days_to_zero)
    
    print(f"\n--- Timewave Zero ({args.mode.upper()} Mode) ---")
    print(f"Zero Date:    {zero_date.strftime('%Y-%m-%d')}")
    print(f"Target Date:  {target_date.strftime('%Y-%m-%d')}")
    print(f"Days to Zero: {days_to_zero:.4f}")
    print(f"Novelty Val:  {novelty:.6e}")
    
    if args.plot:
        start_date = zero_date - timedelta(days=args.years * 365.25)
        generate_graph(zero_date, start_date, args.mode)
    
    if days_to_zero < 0:
        print("\nNote: You are currently in the 'post-historical' era (after the zero point).")
