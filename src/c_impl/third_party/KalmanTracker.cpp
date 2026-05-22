///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"

int KalmanTracker::kf_count = 0;
float KalmanTracker::decay_velocity_factor = 0.2f;

// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 7;
	int measureNum = 4;
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	setIdentity(kf.measurementMatrix);
	setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
	setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kf.errorCovPost, Scalar::all(1));
	
	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}

// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// Velocity decay: if no update recently, slow down to prevent drift
	if (m_time_since_update > 0) {
		float decay = decay_velocity_factor; // Decay factor
		kf.statePost.at<float>(4, 0) *= decay; // vx
		kf.statePost.at<float>(5, 0) *= decay; // vy
		kf.statePost.at<float>(6, 0) *= decay; // vs
	}

	// predict
	Mat p = kf.predict();

	// State propagation: if missing update, the prediction becomes the new state
	if (m_time_since_update > 0) {
		kf.statePre.copyTo(kf.statePost);
		kf.errorCovPre.copyTo(kf.errorCovPost);
	}

	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	StateType predictBox = get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));

	m_history.push_back(predictBox);
	return m_history.back();
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat)
{
	m_time_since_update = 0;
	m_history.clear();
	m_hits += 1;
	m_hit_streak += 1;

	// measurement
	measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	measurement.at<float>(2, 0) = stateMat.area();
	measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

	// update
	kf.correct(measurement);
}

void KalmanTracker::resetVelocity()
{
	kf.statePost.at<float>(4, 0) = 0; // vx
	kf.statePost.at<float>(5, 0) = 0; // vy
	kf.statePost.at<float>(6, 0) = 0; // vs
}

// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	// Clamp s (area) and r (aspect ratio) to avoid NaN
	if (s <= 0) s = 1e-6f;
	if (r <= 0) r = 1e-6f;

	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}

