import { Outlet, Route, Routes } from 'react-router-dom';
import './App.css';
import NavBar from './components/NavBar';
import { AuthProvider } from './contexts/AuthContext';
import SignUpPage from './pages/SignUpPage';
import LoginPage from './pages/LoginPage';
import MainPage from './pages/MainPage';
import QuizCreate from './pages/QuizCreate';
import QuizList from './pages/QuizList';
import QuizView from './pages/QuizView';
import QuizUpdate from './pages/QuizUpdate';

function Layout() {
  return (
    <AuthProvider>
      <NavBar />
      <Outlet />
    </AuthProvider>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<MainPage />} />
        <Route path="auth/signup" element={<SignUpPage />} />
        <Route path="auth/login" element={<LoginPage />} />
        <Route path="member/quiz/create" element={<QuizCreate/>}/>
        <Route path="member/quiz/list" element={<QuizList/>}/>
        <Route path="member/quiz/view" element={<QuizView/>}/>
        <Route path="member/quiz/update" element={<QuizUpdate/>}/>
      </Route>
    </Routes>
  );
}
export default App;