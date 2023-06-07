#include "FeatureSelection.h"

namespace pd::vslam
{
GradientMagnitudeDepth::GradientMagnitudeDepth(
  const std::vector<int> & minGradient, double minDepth, double maxDepth)
: _minDepth(minDepth), _maxDepth(maxDepth)
{
  std::transform(
    minGradient.begin(), minGradient.end(), std::back_inserter(_minGradient2),
    [&](auto g) { return std::pow(g, 2); });
}

Feature2D::VecShPtr GradientMagnitudeDepth::select(Frame::ConstShPtr frame) const
{
  Feature2D::VecShPtr fts;
  fts.reserve(frameRef->width(level) * frameRef->height(level));
  const MatXd gradientMagnitude =
    frameRef->dIx(level).array().pow(2) + frameRef->dIy(level).array().pow(2);
  forEachPixel(gradientMagnitude, [&](int u, int v, double p) {
    if (
      p >= _minGradient2[level] && _minDepth < frameRef->depth(level)(v, u) &&
      frameRef->depth(level)(v, u) < _maxDepth) {
      fts.push_back(std::make_shared<Feature>()).emplace_back(u, v);
    }
  });
  LOG_ODOM(INFO) << "Selected interest points : " << interestPoints.size();

  interestPoints = selectRandomSubset(
    interestPoints, frameRef->height(level), frameCur->width(level), _maxPointsPart);
  LOG_ODOM(INFO) << "Sub-Selected interest points: " << interestPoints.size();
  return interestPoints;
}

UniformSubsampling::UniformSubsampling(const std::vector<double> & maxPoints)
: _maxPoints(maxPoints)
{
}

GridSubsampling::GridSubsampling(const std::vector<double> & cellSize) : _cellSize(cellSize) {}

}  // namespace pd::vslam