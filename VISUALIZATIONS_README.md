# Research Visualization for Koshik Debanath's Profile

This repository contains a legacy research visualization generated using Python and matplotlib. The radar chart is no longer displayed prominently on the portfolio because its normalized impact scores are manually defined and should not be presented as objective academic metrics without a clearer data source.

## 🎯 Overview

The visualization system created a radar chart that highlighted:

- Research domain analysis across multiple dimensions
- Manually normalized publication, citation, and impact-score fields
- Clean, focused presentation without overwhelming detail

## 📊 Generated Visualization

### Research Domains Radar Chart

- **File**: `research_domains_radar.png`
- **Type**: Polar/Radar chart
- **Shows**: Manually normalized fields across 6 research domains
- **Features**: Multi-dimensional analysis with normalized scales, professional styling
- **Current status**: Deprecated from the prominent website experience until the values are tied to documented source data.

## 🛠️ Technical Implementation

### Dependencies

```bash
pip install matplotlib numpy pandas seaborn
```

Required packages:

- `matplotlib` - Core plotting library
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `seaborn` - Statistical visualizations

### Generation Script

```bash
python3 generate_visualizations.py
```

The script automatically:

1. Creates the output directory structure
2. Generates the radar chart visualization (PNG format)
3. Applies consistent styling and color schemes

### Customization

To modify the visualization:

1. **Data Updates**: Edit the data arrays in the radar chart function
2. **Styling**: Modify color schemes, fonts, and layout parameters
3. **Output Format**: Change file formats or dimensions as needed

## 🎨 Design Features

### Color Scheme

- **Primary**: #0078ff (Blue)
- **Secondary**: #00d4ff (Light Blue)
- **Accent**: #FF6B6B (Coral)
- **Success**: #4ECDC4 (Teal)
- **Info**: #45B7D1 (Sky Blue)

### Typography

- **Headers**: Bold, professional fonts
- **Body Text**: Clean, readable sans-serif
- **Consistent sizing hierarchy**

### Layout

- **Responsive design** for all screen sizes
- **Card-based layout** with subtle shadows
- **Hover effects** for enhanced interactivity
- **Professional spacing** and alignment

## 📱 Integration

### HTML Integration

If reintroduced, visualizations should be integrated into the main website via:

- Direct image embedding with responsive classes
- Interactive chart links for enhanced engagement
- Consistent styling with the overall theme

### CSS Styling

Custom CSS provides:

- Smooth hover animations
- Gradient backgrounds
- Enhanced card effects
- Responsive behavior

## 🔄 Maintenance

### Updating Visualizations

1. Modify the data in `generate_visualizations.py`
2. Run the generation script
3. Replace existing image files
4. Update HTML if new visualizations are added

### Adding New Charts

1. Create a new function following the existing pattern
2. Add the function call to the `main()` function
3. Generate the visualization
4. Integrate into the HTML template

## 📈 Future Enhancements

Potential improvements:

- **Real-time data integration** from academic APIs
- **Dynamic chart generation** based on user preferences
- **Export functionality** for presentations and reports
- **Additional chart types** (3D visualizations, treemaps)
- **Mobile-optimized** interactive charts

## 🎯 Use Cases

These visualizations are ideal for:

- **Academic profiles** and research portfolios
- **Grant applications** and funding proposals
- **Conference presentations** and academic talks
- **Collaboration outreach** and networking
- **Research impact assessment** and evaluation

## 📞 Support

For questions or customization requests:

- Review the code comments for implementation details
- Check the matplotlib and plotly documentation for advanced features
- Ensure all dependencies are properly installed

---

*Generated with ❤️ using Python, matplotlib, and plotly*
